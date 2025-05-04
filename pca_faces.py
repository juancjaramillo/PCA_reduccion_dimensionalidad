import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Parámetros
assets_dir = 'assets'  # Carpeta de imágenes locales
k = 10                 # Número de componentes principales
n_plot = 5             # Cuántas imágenes mostrar

# 1) Carga de imágenes locales
paths = [
    os.path.join(assets_dir, f)
    for f in os.listdir(assets_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]
if not paths:
    raise RuntimeError(f"No se encontraron imágenes en '{assets_dir}'")

# Leemos la primera para referencia de tamaño
first = Image.open(paths[0]).convert('L')
w, h = first.size
images = [np.array(first)]
flat = [images[0].flatten()]

# Procesar el resto: convertir y redimensionar si hace falta
for p in paths[1:]:
    img = Image.open(p).convert('L')
    if img.size != (w, h):
        img = img.resize((w, h), Image.LANCZOS)
    arr = np.array(img)
    images.append(arr)
    flat.append(arr.flatten())

# 2) Construir X y centrar
X = np.stack(flat, axis=0)      # (n_samples, d)
mean_face = np.mean(X, axis=0)  # (d,)
X_centered = X - mean_face      # (n_samples, d)
print(f"Cargadas {X.shape[0]} imágenes de tamaño {h}×{w}")

# 3) PCA vía SVD (n_samples ≪ d evita matriz gigante)
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
PC = Vt[:k, :]                 # (k, d)

# 4) Proyección y reconstrucción
Z = X_centered @ PC.T          # (n_samples, k)
X_reconstructed = Z @ PC + mean_face  # (n_samples, d)

# 5) Guardar resultados
os.makedirs('results', exist_ok=True)
for i in range(min(n_plot, X.shape[0])):
    rec = X_reconstructed[i].reshape(h, w).astype(np.uint8)
    Image.fromarray(rec).save(f'results/recon_{i+1}.png')

# 6) Visualizar
plt.figure(figsize=(10, 4))
for i in range(min(n_plot, X.shape[0])):
    # Original
    plt.subplot(2, n_plot, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f"Original {i+1}")
    plt.axis('off')
    # Reconstrucción
    plt.subplot(2, n_plot, n_plot+i+1)
    plt.imshow(X_reconstructed[i].reshape(h, w), cmap='gray')
    plt.title(f"Recon k={k}")
    plt.axis('off')

plt.tight_layout()
plt.show()
