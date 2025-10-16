import argparse
import numpy as np
from PIL import Image

def ler_imagem_cinza(caminho: str) -> np.ndarray:
    img = Image.open(caminho).convert("L")
    return np.array(img, dtype=np.uint8)

def salvar_imagem_cinza(arr: np.ndarray, caminho: str) -> None:
    Image.fromarray(arr.astype(np.uint8), mode="L").save(caminho)

def pad_zero(img: np.ndarray, k: int) -> np.ndarray: # borda zero
    r = k // 2
    if r == 0:
        return img
    return np.pad(img, pad_width=((r, r), (r, r)), mode="constant", constant_values=0)

def erosao_cinza(img: np.ndarray, k: int) -> np.ndarray: #erosão => Ela reduz as regiões brilhantes e remove pequenos brilhos isolados.
    assert k % 2 == 1 and k >= 1, "O tamanho do EE (k) deve ser ímpar e >= 1."
    H, W = img.shape
    r = k // 2
    padded = pad_zero(img, k)
    saida = np.empty_like(img, dtype=np.uint8)

    for i in range(H):
        for j in range(W):
            bloco = padded[i:i+k, j:j+k]
            saida[i, j] = np.min(bloco)
    return saida

def dilatacao_cinza(img: np.ndarray, k: int) -> np.ndarray: #dilatação=> Ela amplia as regiões brilhantes e preenche pequenos buracos escuros.
    assert k % 2 == 1 and k >= 1, "O tamanho do EE (k) deve ser ímpar e >= 1."
    H, W = img.shape
    r = k // 2
    padded = pad_zero(img, k)
    saida = np.empty_like(img, dtype=np.uint8)


    for i in range(H):
        for j in range(W):
            bloco = padded[i:i+k, j:j+k]
            saida[i, j] = np.max(bloco)
    return saida

def abertura_cinza(img: np.ndarray, k: int) -> np.ndarray: #abertura = erosão seguida de dilatação
    e = erosao_cinza(img, k)
    d = dilatacao_cinza(e, k)
    return d

def top_hat(img: np.ndarray, k: int) -> np.ndarray:  #top-hat = img_orginal - abertura
    ab = abertura_cinza(img, k)
    h = img.astype(np.int16) - ab.astype(np.int16)
    h = np.clip(h, 0, 255).astype(np.uint8)
    return h

def main():
    parser = argparse.ArgumentParser(description="Transformada Top-Hat (tons de cinza) com borda zero.")
    parser.add_argument("--entrada", required=True, help="Caminho da imagem de entrada (PNG/JPG).")
    parser.add_argument("--saida", required=True, help="Caminho da imagem de saída.")
    parser.add_argument("--tamanho", type=int, default=5, help="Tamanho do elemento estruturante quadrado (ímpar). Ex: 3, 5, 7.")
    args = parser.parse_args()

    if args.tamanho < 1 or args.tamanho % 2 == 0:
        raise ValueError("O --tamanho do Elemento Estruturante deve ser um número ímpar >= 1 (ex.: 3, 5, 7).")

    img = ler_imagem_cinza(args.entrada)
    h = top_hat(img, args.tamanho)
    salvar_imagem_cinza(h, args.saida)

if __name__ == "__main__":
    main()
