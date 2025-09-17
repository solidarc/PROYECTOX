# extraccion_dinamica.py
# Extracción dinámica de landmarks (cara + pose + manos) con normalización por torso
# y derivadas temporales. Maneja dinámicamente 468 vs 478 puntos de cara (refine).








import os, sys, time, argparse
import cv2
import numpy as np
from datetime import datetime
from mediapipe.python.solutions.holistic import Holistic
# Mantener compatibilidad con tus helpers pero usar nombres en español dentro del script
from zhelpers import draw_keypoints as dibujar_puntos, mediapipe_detection as deteccion_mediapipe

cv2.setNumThreads(0)  # evita deadlocks en algunos entornos

# ---------- utilidades ----------
def asegurar_directorio(ruta): os.makedirs(ruta, exist_ok=True)

def proxima_ruta_indice(directorio, prefijo="bien", ext=".npz"):
    """
    Genera una ruta de archivo del estilo {prefijo}{n:03d}.npz en el directorio dado.
    """
    asegurar_directorio(directorio)
    existentes = [f for f in os.listdir(directorio) if f.startswith(prefijo) and f.endswith(ext)]
    numeros = []
    for f in existentes:
        base = f[len(prefijo):-len(ext)]
        try:
            numeros.append(int(base))
        except:
            pass
    n = (max(numeros) + 1) if numeros else 1
    return os.path.join(directorio, f"{prefijo}{n:03d}{ext}")

def ema(valor_anterior, x, alpha):
    """
    Media móvil exponencial (EMA). Si valor_anterior es None, devuelve x.
    """
    return (alpha * x + (1 - alpha) * valor_anterior) if valor_anterior is not None else x


def interpolar_secuencia(secuencia, longitud_objetivo):
    """
    Interpola en el eje temporal para obtener exactamente 'longitud_objetivo' frames.
    secuencia: array (T, D)
    """
    if len(secuencia) == longitud_objetivo:
        return np.asarray(secuencia, dtype=np.float32)
    secuencia = np.asarray(secuencia, dtype=np.float32)
    x_original = np.linspace(0, 1, len(secuencia))
    x_nuevo = np.linspace(0, 1, longitud_objetivo)
    # Interpolar columna a columna
    return np.stack([np.interp(x_nuevo, x_original, secuencia[:, d]) for d in range(secuencia.shape[1])], axis=1)
 

def apilar_con_derivadas(secuencia):
    """
    Devuelve [pos, vel, acc] concatenados en el eje de features.
    secuencia: (T, D) -> salida: (T, D*3)
    """
    vel = np.diff(secuencia, axis=0, prepend=secuencia[0:1])
    acc = np.diff(vel, axis=0, prepend=vel[0:1])
    return np.concatenate([secuencia, vel, acc], axis=1)  # (T, D*3)


# ---------- extracción y normalización ----------
def extraer_puntos(resultados, cara_refinada_por_defecto=True):
    """
    Devuelve:
      vector: np.ndarray shape (D,) con [face(xyz), pose(xyz+vis), lh(xyz), rh(xyz)]
      cant_puntos_cara: cantidad de puntos de cara usados (468 o 478)
    """
    def lms_a_xyz(lms, incluir_vis=False):
        if lms is None:
            return None
        salida = []
        for lm in lms.landmark:
            if incluir_vis:
                salida.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:
                salida.extend([lm.x, lm.y, lm.z])
        return np.array(salida, dtype=np.float32)

    cara_arr = lms_a_xyz(resultados.face_landmarks, incluir_vis=False)
    if cara_arr is not None:
        cant_puntos_cara = cara_arr.shape[0] // 3  # 468 o 478
    else:
        # Si falla la cara, asumimos según refine
        cant_puntos_cara = 478 if cara_refinada_por_defecto else 468
        cara_arr = np.zeros(cant_puntos_cara * 3, np.float32)

    pose_arr = lms_a_xyz(resultados.pose_landmarks, incluir_vis=True)
    if pose_arr is None:
        pose_arr = np.zeros(33 * 4, np.float32)

    mano_izq_arr = lms_a_xyz(resultados.left_hand_landmarks, incluir_vis=False)
    if mano_izq_arr is None:
        mano_izq_arr = np.zeros(21 * 3, np.float32)

    mano_der_arr = lms_a_xyz(resultados.right_hand_landmarks, incluir_vis=False)
    if mano_der_arr is None:
        mano_der_arr = np.zeros(21 * 3, np.float32)

    vector = np.concatenate([cara_arr, pose_arr, mano_izq_arr, mano_der_arr], axis=0)
    return vector, cant_puntos_cara


def normalizar_por_torso(secuencia, cant_puntos_cara, valor_clamp=5.0):
    """
    Normaliza (x,y,z) restando centro de hombros y escalando por distancia entre hombros.
    No modifica las columnas de 'visibility' del pose.
    Aplica clamp de outliers a [-valor_clamp, valor_clamp].
    secuencia: (T, D_base)
    """
    off_cara = cant_puntos_cara * 3
    d_pose = 33 * 4
    if secuencia.shape[1] < off_cara + d_pose:
        return secuencia

    T, D = secuencia.shape
    salida = secuencia.copy()

    # Pose: (33, 4) = (x,y,z,visibility)
    pose = salida[:, off_cara:off_cara + d_pose].reshape(T, 33, 4)
    hombro_izq = pose[:, 11, :3]
    hombro_der = pose[:, 12, :3]
    centro = 0.5 * (hombro_izq + hombro_der)  # (T, 3)
    escala = np.linalg.norm(hombro_izq - hombro_der, axis=1)[:, None] + 1e-6  # (T,1)

    # Máscara de columnas visibility del bloque pose
    cols_vis = np.zeros(D, dtype=bool)
    for k in range(33):
        cols_vis[off_cara + k*4 + 3] = True

    # Recorremos de a (x,y,z) y normalizamos, saltando columnas 'visibility'
    for t in range(T):
        i = 0
        while i < D:
            if cols_vis[i]:
                i += 1
                continue
            # normalizamos triple (x,y,z) si existen
            for j in range(3):
                if i + j < D and not cols_vis[i + j]:
                    salida[t, i + j] = (salida[t, i + j] - centro[t, j]) / escala[t, 0]
            i += 3

    # clamp para evitar outliers extremos
    salida = np.clip(salida, -valor_clamp, valor_clamp)
    return salida


def guardar_clip(ruta, secuencia, longitud_secuencia, fps, etiqueta, cant_puntos_cara,
                 id_sujeto=None, etiqueta_camara=None, iluminacion=None):
    """
    Guarda:
      - keypoints: (T, D_base_norm)  [ya normalizados por torso]
      - features:  (T, D_base*3)     [pos+vel+acc]
      - meta: fps, etiqueta, longitud_secuencia, cant_puntos_cara, created, y opcionales id_sujeto/etiqueta_camara/iluminacion
    """
    arr = np.stack(secuencia, axis=0)                     # (T0, D_base)
    arr = normalizar_por_torso(arr, cant_puntos_cara)     # (T0, D_base)
    arr = interpolar_secuencia(arr, longitud_secuencia)   # (T, D_base)
    feats = apilar_con_derivadas(arr)                     # (T, D_base*3)

    meta = {
        "fps": np.array([fps], dtype=np.int32),
        "etiqueta": np.array([etiqueta.upper()]),
        "longitud_secuencia": np.array([longitud_secuencia], dtype=np.int32),
        "cant_puntos_cara": np.array([cant_puntos_cara], dtype=np.int32),
        "creado": np.array([datetime.now().isoformat()])
    }
    if id_sujeto is not None:
        meta["id_sujeto"] = np.array([str(id_sujeto)])
    if etiqueta_camara is not None:
        meta["etiqueta_camara"] = np.array([str(etiqueta_camara)])
    if iluminacion is not None:
        meta["iluminacion"] = np.array([str(iluminacion)])

    np.savez_compressed(
        ruta,
        keypoints=arr.astype(np.float32),
        features=feats.astype(np.float32),
        **meta
    )


# ---------- app ----------
def main():
    ap = argparse.ArgumentParser("LSA extractor dinámico (safe, refine-aware)")
    # ---- Flags en español
    ap.add_argument("--etiqueta", default="BIEN")
    ap.add_argument("--directorio_salida", default="dataset")
    ap.add_argument("--prefijo", default="bien")
    ap.add_argument("--longitud_secuencia", type=int, default=48)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--suavizado", type=float, default=0.35)
    ap.add_argument("--camara", type=int, default=1)
    ap.add_argument("--backend", choices=["auto","dshow","msmf","v4l2"], default="auto")
    ap.add_argument("--ancho", type=int, default=640)
    ap.add_argument("--alto", type=int, default=480)
    ap.add_argument("--inicio_auto", action="store_true")
    ap.add_argument("--fin_auto", action="store_true")
    ap.add_argument("--frames_inicio_presencia", type=int, default=5)
    ap.add_argument("--frames_fin_ausencia", type=int, default=10)
    ap.add_argument("--refinar_cara", action="store_true", help="Activa refine_face_landmarks=True (usa 478 puntos).")
    ap.add_argument("--id_sujeto", default=None)
    ap.add_argument("--etiqueta_camara", default=None)
    ap.add_argument("--iluminacion", default=None)
    ap.add_argument("--depuracion", action="store_true")
    # ---- Compatibilidad hacia atrás (flags originales en inglés → mismo destino)
    ap.add_argument("--label", dest="etiqueta")
    ap.add_argument("--out_dir", dest="directorio_salida")
    ap.add_argument("--prefix", dest="prefijo")
    ap.add_argument("--seq_len", dest="longitud_secuencia", type=int)
    ap.add_argument("--camera", dest="camara", type=int)
    ap.add_argument("--width", dest="ancho", type=int)
    ap.add_argument("--height", dest="alto", type=int)
    ap.add_argument("--autostart", dest="inicio_auto", action="store_true")
    ap.add_argument("--autostop", dest="fin_auto", action="store_true")
    ap.add_argument("--presence_start_frames", dest="frames_inicio_presencia", type=int)
    ap.add_argument("--absence_stop_frames", dest="frames_fin_ausencia", type=int)
    ap.add_argument("--refine_face", dest="refinar_cara", action="store_true")
    ap.add_argument("--subject_id", dest="id_sujeto")
    ap.add_argument("--camera_tag", dest="etiqueta_camara")
    ap.add_argument("--lighting", dest="iluminacion")
    ap.add_argument("--debug", dest="depuracion", action="store_true")

    args = ap.parse_args()

    # backend cámara
    flag_backend = 0
    if args.backend == "dshow": flag_backend = cv2.CAP_DSHOW
    elif args.backend == "msmf": flag_backend = cv2.CAP_MSMF
    elif args.backend == "v4l2": flag_backend = cv2.CAP_V4L2

    cap = cv2.VideoCapture(args.camara, flag_backend) if flag_backend != 0 else cv2.VideoCapture(args.camara)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara {args.camara} (backend={args.backend}).")
        print("Sugerencias: probar otro --camara (0/1) y en Windows usar --backend dshow.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.ancho)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.alto)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    etiqueta = args.etiqueta.upper()
    directorio_guardado = os.path.join(args.directorio_salida, etiqueta)
    ruta_guardado = proxima_ruta_indice(directorio_guardado, prefijo=args.prefijo.lower(), ext=".npz")

    # Modelo Holistic
    holistic = Holistic(
        static_image_mode=False,
        model_complexity=1,             # baja complejidad = más estable en equipos modestos
        smooth_landmarks=True,
        refine_face_landmarks=bool(args.refinar_cara),
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("Controles: [r]=grabar  [s]=guardar y salir  [ESC/x]=salir sin guardar")
    print(f"Etiqueta: {etiqueta}  → próximo archivo: {os.path.basename(ruta_guardado)}")
    if args.inicio_auto: print(f"Auto-REC ON ({args.frames_inicio_presencia} frames con manos)")
    if args.fin_auto:    print(f"Auto-STOP ON ({args.frames_fin_ausencia} frames sin manos)")
    print(f"refine_face_landmarks = {bool(args.refinar_cara)} (cara será 478 si está activo)")

    grabando = False
    secuencia = []
    ema_anterior = None
    frames_grabados = 0
    conteo_presencia = 0
    conteo_ausencia  = 0
    cant_puntos_cara_usada = None  # se fija cuando llega el primer frame válido
    t0 = time.time()
    frames_mostrados = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] No se pudo leer frame.")
                break

            frame = cv2.flip(frame, 1)

            # ---- pipeline con helpers ----
            resultados = deteccion_mediapipe(frame, holistic)  # helper
            imagen = frame.copy()
            dibujar_puntos(imagen, resultados)                 # helper

            # estados para overlay
            cara_ok  = resultados.face_landmarks is not None
            mano_i_ok = resultados.left_hand_landmarks is not None
            mano_d_ok = resultados.right_hand_landmarks is not None
            pose_ok = resultados.pose_landmarks is not None

            # presencia/ausencia según manos
            if mano_i_ok or mano_d_ok:
                conteo_presencia += 1; conteo_ausencia = 0
            else:
                conteo_ausencia += 1; conteo_presencia = 0

            # auto-REC
            if args.inicio_auto and not grabando and conteo_presencia >= args.frames_inicio_presencia:
                print("Auto-REC: manos detectadas.")
                grabando = True; secuencia = []; frames_grabados = 0

            # features por frame
            vector, cant_puntos_cara = extraer_puntos(resultados, cara_refinada_por_defecto=bool(args.refinar_cara))
            # fijamos cant_puntos_cara de la secuencia al primer valor confiable
            if cant_puntos_cara_usada is None:
                cant_puntos_cara_usada = cant_puntos_cara
            elif cant_puntos_cara != cant_puntos_cara_usada:
                # no debería ocurrir; avisamos y seguimos usando el primero
                print(f"[WARN] cant_puntos_cara cambió {cant_puntos_cara_usada}→{cant_puntos_cara}; mantengo {cant_puntos_cara_usada}")

            ema_anterior = ema(ema_anterior, vector, args.suavizado)
            vector_suavizado = ema_anterior.copy()

            if grabando:
                secuencia.append(vector_suavizado)
                frames_grabados += 1

            # auto-STOP
            if args.fin_auto and grabando and conteo_ausencia >= args.frames_fin_ausencia:
                print("Auto-STOP: manos ausentes, guardando clip...")
                if len(secuencia) > 4:
                    guardar_clip(
                        ruta_guardado, secuencia, args.longitud_secuencia, args.fps, etiqueta,
                        cant_puntos_cara_usada if cant_puntos_cara_usada is not None else (478 if args.refinar_cara else 468),
                        id_sujeto=args.id_sujeto, etiqueta_camara=args.etiqueta_camara, iluminacion=args.iluminacion
                    )
                    # reporte de dimensiones
                    D_base = (cant_puntos_cara_usada if cant_puntos_cara_usada is not None else (478 if args.refinar_cara else 468)) * 3 + 33*4 + 21*3 + 21*3
                    print(f"[OK] Guardado: {ruta_guardado} | T={len(secuencia)} | D_base={D_base} -> features={D_base*3}")
                else:
                    print("[WARN] Clip demasiado corto, descartado.")
                break

            # overlays
            color = (0,255,0) if grabando else (255,255,255)
            cv2.putText(imagen, f"{etiqueta} -> {os.path.basename(ruta_guardado)}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(imagen, "REC" if grabando else "LISTO", (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255) if grabando else (200,200,200), 2)
            cv2.putText(imagen, f"Frames: {frames_grabados if grabando else 0}", (10, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            estado = f"CARA:{'Y' if cara_ok else 'N'}  MI:{'Y' if mano_i_ok else 'N'}  MD:{'Y' if mano_d_ok else 'N'}  POSE:{'Y' if pose_ok else 'N'}"
            cv2.putText(imagen, estado, (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)

            cv2.imshow('Extracción LSA (segura, refine-aware)', imagen)
            frames_mostrados += 1

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord('r') and not grabando:
                print("REC manual...")
                grabando = True; secuencia = []; frames_grabados = 0
            elif tecla == ord('s'):
                if grabando and len(secuencia) > 4:
                    print("Guardando clip...")
                    guardar_clip(
                        ruta_guardado, secuencia, args.longitud_secuencia, args.fps, etiqueta,
                        cant_puntos_cara_usada if cant_puntos_cara_usada is not None else (478 if args.refinar_cara else 468),
                        id_sujeto=args.id_sujeto, etiqueta_camara=args.etiqueta_camara, iluminacion=args.iluminacion
                    )
                    D_base = (cant_puntos_cara_usada if cant_puntos_cara_usada is not None else (478 if args.refinar_cara else 468)) * 3 + 33*4 + 21*3 + 21*3
                    print(f"[OK] Guardado: {ruta_guardado} | T={len(secuencia)} | D_base={D_base} -> features={D_base*3}")
                else:
                    print("[WARN] Nada que guardar (o clip muy corto).")
                break
            elif tecla in (27, ord('x')):  # ESC o 'x'
                print("Salida sin guardar.")
                break

            # debug FPS
            if args.depuracion and frames_mostrados % 30 == 0:
                dt = max(time.time() - t0, 1e-6)
                fps_med = frames_mostrados / dt
                print(f"[DEBUG] FPS~{fps_med:.1f} | grabando={grabando} | presencia={conteo_presencia} ausencia={conteo_ausencia}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            holistic.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
