import cv2
import pandas as pd
import numpy as np
import math
import sys
from pathlib import Path

# Função para listar todos os arquivos de imagem no diretório especificado
def listar_imagens(diretorio):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    arquivos = []
    for ext in exts:
        arquivos.extend(Path(diretorio).glob(ext))
    return [str(arq) for arq in sorted(arquivos)]

# Função de pré-processamento da imagem
def preprocessar_imagem(imagem_bgr):
    imagem_gray = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imagem_gray = clahe.apply(imagem_gray)
    imagem_gray = cv2.bilateralFilter(imagem_gray, d=5, sigmaColor=75, sigmaSpace=75)
    bordas = cv2.Canny(imagem_gray, threshold1=50, threshold2=150)
    imagem_processada = cv2.cvtColor(imagem_gray, cv2.COLOR_GRAY2BGR)
    mask = bordas != 0
    imagem_processada[mask] = [0, 0, 255]
    return imagem_processada

# Função para redesenhar a imagem com anotações
def redesenhar_imagem(state):
    img = state["base_img"].copy()
    if len(state["calib_points"]) == 1 and not state["calibrated"]:
        cv2.circle(img, state["calib_points"][0], 5, (255, 255, 0), -1)
    if state["calibrated"]:
        pt1, pt2 = state["calib_points"]
        cv2.circle(img, pt1, 5, (255, 255, 0), -1)
        cv2.circle(img, pt2, 5, (255, 255, 0), -1)
        cv2.line(img, pt1, pt2, (255, 255, 0), 2)
        dist_cm = state["scale"] * state["calib_dist_px"]
        mx, my = (pt1[0]+pt2[0])//2, (pt1[1]+pt2[1])//2
        cv2.putText(img, f"{dist_cm:.2f} cm", (mx+5, my-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    for p1, p2, dist_px, dist_cm in state["measurements"]:
        cv2.circle(img, p1, 5, (0, 255, 0), -1)
        cv2.circle(img, p2, 5, (0, 255, 0), -1)
        cv2.line(img, p1, p2, (0, 255, 0), 2)
        mx, my = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2
        cv2.putText(img, f"{dist_cm:.2f} cm", (mx+5, my-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
        cv2.putText(img, f"{dist_cm:.2f} cm", (mx+5, my-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    if state["calibrated"] and state["first_point"] is not None:
        cv2.circle(img, state["first_point"], 5, (0, 165, 255), -1)
    return img

# Callback de mouse para eventos de clique
def evento_mouse(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN:
        if not state["calibrated"]:
            if len(state["calib_points"]) < 2:
                state["calib_points"].append((x, y))
                if len(state["calib_points"]) == 2:
                    pt1, pt2 = state["calib_points"]
                    dx, dy = pt2[0]-pt1[0], pt2[1]-pt1[1]
                    dist_px = math.hypot(dx, dy)
                    state["calib_dist_px"] = dist_px
                    val = input(f"Distância real (cm) entre calibragem em {state['img_name']}: ")
                    try:
                        real_val = float(val.replace(',', '.'))
                    except:
                        real_val = 1.0
                    state["scale"] = real_val / dist_px
                    state["calibrated"] = True
            state["display_img"] = redesenhar_imagem(state)
        else:
            if state["first_point"] is None:
                state["first_point"] = (x, y)
            else:
                p1 = state["first_point"]
                p2 = (x, y)
                dx, dy = p2[0]-p1[0], p2[1]-p1[1]
                dist_px = math.hypot(dx, dy)
                dist_cm = dist_px * state["scale"]
                state["measurements"].append((p1, p2, dist_px, dist_cm))
                state["first_point"] = None
            state["display_img"] = redesenhar_imagem(state)

# Processa todas as imagens no diretório dado e salva resultados
def processar_imagens(input_dir, output_csv):
    arquivos = listar_imagens(input_dir)
    if not arquivos:
        print("Nenhuma imagem encontrada.")
        return
    resultados = []
    cv2.namedWindow("Imagem", cv2.WINDOW_NORMAL)
    for caminho in arquivos:
        img = cv2.imread(caminho)
        if img is None:
            print(f"Erro ao carregar {caminho}")
            continue
        img_proc = preprocessar_imagem(img)
        state = {"img_name": Path(caminho).name,
                 "base_img": img_proc,
                 "display_img": img_proc.copy(),
                 "calib_points": [],
                 "calibrated": False,
                 "calib_dist_px": None,
                 "scale": None,
                 "first_point": None,
                 "measurements": []}
        cv2.setMouseCallback("Imagem", evento_mouse, state)
        print(f"Processando: {state['img_name']}")
        print("Clique 2 pontos para calibração, informe valor cm. Depois pares para medir.")
        print("Teclas: u=undo, r=reset, n=próxima, q=quit.")
        while True:
            cv2.imshow("Imagem", state["display_img"])
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                cv2.destroyWindow("Imagem")
                df = pd.DataFrame(resultados, columns=["Arquivo","Ponto 1","Ponto 2","Distância em Pixels","Distância Real (cm)"])
                df.to_csv(output_csv, index=False)
                print(f"Resultados salvos em {output_csv}")
                return
            if key == ord('n'):
                break
            if key == ord('u'):
                if state["first_point"] is not None:
                    state["first_point"] = None
                elif not state["calibrated"] and len(state["calib_points"]) == 1:
                    state["calib_points"].clear()
                elif state["measurements"]:
                    state["measurements"].pop()
                state["display_img"] = redesenhar_imagem(state)
            if key == ord('r'):
                state.update({"calib_points": [], "calibrated": False,
                              "calib_dist_px": None, "scale": None,
                              "first_point": None, "measurements": []})
                state["display_img"] = state["base_img"].copy()
        for p1, p2, dist_px, dist_cm in state["measurements"]:
            resultados.append([state["img_name"],
                               f"({p1[0]}, {p1[1]})",
                               f"({p2[0]}, {p2[1]})",
                               round(dist_px,2), round(dist_cm,2)])
    cv2.destroyWindow("Imagem")
    if resultados:
        df = pd.DataFrame(resultados, columns=["Arquivo","Ponto 1","Ponto 2","Distância em Pixels","Distância Real (cm)"])
        df.to_csv(output_csv, index=False)
        print(f"CSV gerado: {output_csv}")
    else:
        print("Nenhuma medição realizada.")

# Entrada principal
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python script.py <diretório_imagens> <arquivo_saida.csv>")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_csv = sys.argv[2]
    processar_imagens(input_dir, output_csv)
