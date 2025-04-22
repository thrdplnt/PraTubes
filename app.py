import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import logging
import time
from sklearn.preprocessing import LabelEncoder
from deepface import DeepFace
import tempfile

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dapatkan direktori kerja saat ini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_FILE = os.path.join(BASE_DIR, 'dataset_metadata.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'ethnic_model.pth')
SIAMESE_MODEL_PATH = os.path.join(BASE_DIR, 'siamese_model.h5')
CONSENT_DIR = os.path.join(BASE_DIR, 'consents')
os.makedirs(CONSENT_DIR, exist_ok=True)

# Inisialisasi Session State untuk siamese_base_model
if 'siamese_base_model' not in st.session_state:
    st.session_state.siamese_base_model = None

# Coba impor MTCNN
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
    st.success("MTCNN berhasil diimpor!")
except ImportError as e:
    MTCNN_AVAILABLE = False
    st.error(f"Gagal mengimpor MTCNN: {str(e)}")
    st.warning("MTCNN tidak tersedia. Menggunakan Haar Cascade.")

# Coba impor RetinaFace
try:
    from retinaface import RetinaFace
    RETINAFACE_AVAILABLE = True
    st.success("RetinaFace berhasil diimpor!")
except ImportError as e:
    RETINAFACE_AVAILABLE = False
    st.error(f"Gagal mengimpor RetinaFace: {str(e)}")
    st.warning("RetinaFace tidak tersedia. Menggunakan MTCNN atau Haar Cascade.")

# Coba impor imgaug
try:
    import imgaug.augmenters as iaa
    IMGAUG_AVAILABLE = True
except ImportError:
    IMGAUG_AVAILABLE = False
    st.warning("imgaug tidak tersedia. Data augmentation dinonaktifkan.")

# Inisialisasi FaceNet
FACENET_AVAILABLE = True
try:
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
    if torch.cuda.is_available():
        facenet_model = facenet_model.cuda()
    st.success("FaceNet model berhasil diimpor!")
except Exception as e:
    FACENET_AVAILABLE = False
    st.error(f"Gagal mengimpor FaceNet: {str(e)}")
    st.warning("Face Similarity (FaceNet) tidak tersedia.")

# Inisialisasi ArcFace
ARCFACE_AVAILABLE = True
try:
    arcface_model = models.resnet50(weights='IMAGENET1K_V1')
    arcface_model.fc = nn.Linear(arcface_model.fc.in_features, 512)
    if torch.cuda.is_available():
        arcface_model = arcface_model.cuda()
    arcface_model.eval()
    st.success("ArcFace model (ResNet50) berhasil diimpor!")
except Exception as e:
    ARCFACE_AVAILABLE = False
    st.error(f"Gagal mengimpor ArcFace: {str(e)}")
    st.warning("Face Similarity (ArcFace) tidak tersedia.")

# Fungsi preprocessing gambar
def preprocess_image_for_facenet(image, size=(160, 160)):
    try:
        image = cv2.resize(image, size)
        image = image.astype(np.float32)
        image = (image - 127.5) / 128.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image).unsqueeze(0)
        if torch.cuda.is_available():
            image = image.cuda()
        return image
    except Exception as e:
        logging.error(f"Error preprocessing for FaceNet: {e}")
        return None

def preprocess_image_for_siamese(image, size=(160, 160)):
    try:
        image = cv2.resize(image, size)
        image = image.astype(np.float32) / 255.0
        return image
    except Exception as e:
        logging.error(f"Error preprocessing for Siamese: {e}")
        return None

def preprocess_image_for_arcface(image, size=(224, 224)):
    try:
        image = cv2.resize(image, size)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0)
        if torch.cuda.is_available():
            image = image.cuda()
        return image
    except Exception as e:
        logging.error(f"Error preprocessing for ArcFace: {e}")
        return None

# Fungsi mendapatkan embedding wajah
def get_face_embedding_facenet(image):
    if not FACENET_AVAILABLE:
        st.error("FaceNet tidak tersedia.")
        return None
    try:
        processed_image = preprocess_image_for_facenet(image)
        if processed_image is None:
            return None
        with torch.no_grad():
            embedding = facenet_model(processed_image).cpu().numpy()
        return embedding.flatten()
    except Exception as e:
        logging.error(f"Gagal mendapatkan embedding FaceNet: {e}")
        return None

def get_face_embedding_siamese(image, siamese_model):
    try:
        processed_image = preprocess_image_for_siamese(image)
        if processed_image is None:
            return None
        processed_image = np.expand_dims(processed_image, axis=0)
        embedding = siamese_model.predict(processed_image, verbose=0)
        return embedding.flatten()
    except Exception as e:
        logging.error(f"Gagal mendapatkan embedding Siamese: {e}")
        return None

def get_face_embedding_arcface(image):
    if not ARCFACE_AVAILABLE:
        st.error("ArcFace tidak tersedia.")
        return None
    try:
        processed_image = preprocess_image_for_arcface(image)
        if processed_image is None:
            return None
        with torch.no_grad():
            embedding = arcface_model(processed_image).cpu().numpy()
        return embedding.flatten()
    except Exception as e:
        logging.error(f"Gagal mendapatkan embedding ArcFace: {e}")
        return None

# Fungsi deteksi wajah
def preprocess_image_for_detection(image):
    try:
        scale_factor = 2
        resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        if len(resized_image.shape) == 3:
            gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            resized_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        else:
            resized_image = cv2.equalizeHist(resized_image)
        return resized_image, scale_factor
    except Exception as e:
        logging.error(f"Error preprocessing for detection: {e}")
        return image, 1

def detect_faces_haar(image):
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        preprocessed_image, scale_factor = preprocess_image_for_detection(image)
        gray = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))
        return [(int(x/scale_factor), int(y/scale_factor), int(w/scale_factor), int(h/scale_factor)) for (x, y, w, h) in faces]
    except Exception as e:
        logging.error(f"Error detecting faces with Haar: {e}")
        return []

def detect_faces_mtcnn(image):
    if not MTCNN_AVAILABLE:
        return detect_faces_haar(image)
    try:
        preprocessed_image, scale_factor = preprocess_image_for_detection(image)
        detector = MTCNN(min_face_size=20, steps_threshold=[0.6, 0.7, 0.7])
        faces = detector.detect_faces(preprocessed_image)
        return [
            (int(face['box'][0]/scale_factor), int(face['box'][1]/scale_factor),
             int(face['box'][2]/scale_factor), int(face['box'][3]/scale_factor))
            for face in faces
        ]
    except Exception as e:
        logging.error(f"Error detecting faces with MTCNN: {e}")
        return detect_faces_haar(image)

def detect_faces_retina(image):
    if not RETINAFACE_AVAILABLE:
        return detect_faces_mtcnn(image)
    try:
        preprocessed_image, scale_factor = preprocess_image_for_detection(image)
        faces = RetinaFace.detect_faces(preprocessed_image)
        adjusted_faces = []
        for key in faces.keys():
            face = faces[key]
            x, y, w, h = face['facial_area']
            adjusted_faces.append((
                int(x/scale_factor), int(y/scale_factor),
                int(w/scale_factor) - int(x/scale_factor),
                int(h/scale_factor) - int(y/scale_factor)
            ))
        return adjusted_faces
    except Exception as e:
        logging.error(f"Error detecting faces with RetinaFace: {e}")
        return detect_faces_mtcnn(image)

def detect_faces(image, method='retina'):
    start_time = time.time()
    if method == 'retina' and RETINAFACE_AVAILABLE:
        faces = detect_faces_retina(image)
    elif method == 'mtcnn' and MTCNN_AVAILABLE:
        faces = detect_faces_mtcnn(image)
    else:
        faces = detect_faces_haar(image)
    elapsed_time = time.time() - start_time
    logging.info(f"Face detection ({method}) took {elapsed_time:.2f} seconds")
    return faces

# Fungsi augmentation
def augment_rotation(image):
    if not IMGAUG_AVAILABLE:
        logging.warning("Rotasi tidak dilakukan karena imgaug tidak tersedia.")
        return image
    seq = iaa.Affine(rotate=(-15, 15))
    return seq.augment_image(image)

def augment_flip(image):
    if not IMGAUG_AVAILABLE:
        logging.warning("Horizontal flip tidak dilakukan karena imgaug tidak tersedia.")
        return image
    seq = iaa.Fliplr(1.0)
    return seq.augment_image(image)

def augment_brightcontrast(image):
    if not IMGAUG_AVAILABLE:
        logging.warning("Perubahan brightness/contrast tidak dilakukan karena imgaug tidak tersedia.")
        return image
    seq = iaa.Sequential([
        iaa.Multiply((0.8, 1.2)),
        iaa.ContrastNormalization((0.8, 1.2))
    ])
    return seq.augment_image(image)

def augment_gaussian_noise(image):
    if not IMGAUG_AVAILABLE:
        logging.warning("Penambahan noise Gaussian tidak dilakukan karena imgaug tidak tersedia.")
        return image
    seq = iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))
    return seq.augment_image(image)

# Visualisasi probabilitas gender
def visualize_gender_probabilities(probabilities, classes=['Male', 'Female']):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=probabilities, y=classes)
    plt.xlabel("Probabilitas")
    plt.title("Distribusi Probabilitas Gender")
    st.pyplot(plt)
    plt.clf()

# Inisialisasi file CSV metadata
def init_metadata_csv():
    expected_columns = ['path_gambar', 'nama', 'suku', 'ekspresi', 'sudut', 'pencahayaan']
    if not os.path.exists(METADATA_FILE):
        df = pd.DataFrame(columns=expected_columns)
        try:
            df.to_csv(METADATA_FILE, index=False)
            logging.info("Metadata CSV initialized")
        except PermissionError as e:
            st.error(f"Gagal membuat file metadata: {e}")
    else:
        try:
            df = pd.read_csv(METADATA_FILE)
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = ''
            df = df[expected_columns]
            df.to_csv(METADATA_FILE, index=False)
            logging.info("Metadata CSV updated")
        except PermissionError as e:
            st.error(f"Gagal membaca/menulis file metadata: {e}")

# Tambahkan data ke CSV
def add_to_metadata(path, nama, suku, ekspresi, sudut, pencahayaan):
    try:
        nama = nama.capitalize()
        suku = suku.capitalize()
        ekspresi = ekspresi.lower()
        sudut = sudut.replace("miring 45° kiri", "miring-45-kiri").replace("miring 45° kanan", "miring-45-kanan").lower()
        pencahayaan = pencahayaan.lower()
        df = pd.DataFrame([{
            'path_gambar': path,
            'nama': nama,
            'suku': suku,
            'ekspresi': ekspresi,
            'sudut': sudut,
            'pencahayaan': pencahayaan
        }])
        df.to_csv(METADATA_FILE, mode='a', header=False, index=False)
        logging.info(f"Added to metadata: {path}")
    except PermissionError as e:
        st.error(f"Gagal menulis ke file metadata: {e}")

# Simpan formulir persetujuan
def save_consent_form(nama, suku):
    consent_text = f"""
    Formulir Persetujuan Pengambilan Foto
    Nama: {nama}
    Suku/Etnis: {suku}
    Saya menyetujui penggunaan foto wajah saya untuk keperluan akademis dalam proyek pengenalan wajah.
    Tanggal: {time.strftime('%Y-%m-%d')}
    """
    consent_path = os.path.join(CONSENT_DIR, f"consent_{nama}_{suku}.txt")
    try:
        with open(consent_path, 'w') as f:
            f.write(consent_text)
        logging.info(f"Consent form saved: {consent_path}")
        return True
    except Exception as e:
        st.error(f"Gagal menyimpan formulir persetujuan: {e}")
        return False

# Split dataset
def split_dataset():
    if not os.path.exists(METADATA_FILE):
        st.error("File metadata tidak ditemukan.")
        return

    df = pd.read_csv(METADATA_FILE)
    df = df[df['path_gambar'].str.contains(r'dataset_kel_9[\\/]', na=False)]

    if len(df) == 0:
        st.warning("Tidak ada gambar di metadata dari dataset_kel_9.")
        return

    grouped = df.groupby(['nama', 'suku'])
    train_dir = 'train_dataset_kel_9'
    val_dir = 'val_dataset_kel_9'
    test_dir = 'test_dataset_kel_9'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_metadata = []
    val_metadata = []
    test_metadata = []

    total_train = 0
    total_val = 0
    total_test = 0

    for (nama, suku), group in grouped:
        image_paths = group['path_gambar'].tolist()
        total_images = len(image_paths)

        num_train = max(1, int(total_images * 0.7))
        num_val = max(1, int(total_images * 0.15))
        num_test = total_images - num_train - num_val

        if num_test < 1 and total_images >= 3:
            num_val -= 1
            num_test = 1
        elif num_test < 1:
            num_val = total_images - num_train
            num_test = 0

        np.random.shuffle(image_paths)

        train_paths = image_paths[:num_train]
        val_paths = image_paths[num_train:num_train + num_val]
        test_paths = image_paths[num_train + num_val:]

        total_train += len(train_paths)
        total_val += len(val_paths)
        total_test += len(test_paths)

        def copy_images(image_paths, target_dir, split_metadata):
            for img_path in image_paths:
                row = df[df['path_gambar'] == img_path].iloc[0]
                target_path = os.path.join(target_dir, nama, suku)
                os.makedirs(target_path, exist_ok=True)
                new_img_path = os.path.join(target_path, os.path.basename(img_path))
                shutil.copy(img_path, new_img_path)
                split_metadata.append({
                    'path_gambar': new_img_path,
                    'nama': nama,
                    'suku': suku,
                    'ekspresi': row['ekspresi'],
                    'sudut': row['sudut'],
                    'pencahayaan': row['pencahayaan']
                })

        copy_images(train_paths, train_dir, train_metadata)
        copy_images(val_paths, val_dir, val_metadata)
        copy_images(test_paths, test_dir, test_metadata)

        st.write(f"Subjek: {nama} ({suku})")
        st.write(f"- Total gambar: {total_images}")
        st.write(f"- Training: {len(train_paths)} ({len(train_paths)/total_images*100:.1f}%)")
        st.write(f"- Validasi: {len(val_paths)} ({len(val_paths)/total_images*100:.1f}%)")
        st.write(f"- Testing: {len(test_paths)} ({len(test_paths)/total_images*100:.1f}%)")
        st.write("---")

    pd.DataFrame(train_metadata).to_csv("train_metadata.csv", index=False)
    pd.DataFrame(val_metadata).to_csv("val_metadata.csv", index=False)
    pd.DataFrame(test_metadata).to_csv("test_metadata.csv", index=False)

    st.success(f"Dataset dibagi: {total_train} training, {total_val} validation, {total_test} testing.")

# Siamese Network dengan Triplet Loss
def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[:, :128], y_pred[:, 128:256], y_pred[:, 256:]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss

def create_base_network(input_shape=(160, 160, 3)):
    model = tf.keras.Sequential([
        Conv2D(64, (10, 10), activation='relu', input_shape=input_shape),
        MaxPooling2D(),
        Conv2D(128, (7, 7), activation='relu'),
        MaxPooling2D(),
        Conv2D(128, (4, 4), activation='relu'),
        MaxPooling2D(),
        Conv2D(256, (4, 4), activation='relu'),
        Flatten(),
        Dense(4096, activation='relu'),
        Dense(128, activation='sigmoid')
    ])
    return model

def create_siamese_model(input_shape=(160, 160, 3)):
    base_network = create_base_network(input_shape)
    input_a = Input(shape=input_shape)
    input_p = Input(shape=input_shape)
    input_n = Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_p = base_network(input_p)
    processed_n = base_network(input_n)
    merged = Lambda(lambda x: tf.concat(x, axis=-1))([processed_a, processed_p, processed_n])
    model = Model([input_a, input_p, input_n], merged)
    return model, base_network

def train_siamese_model():
    if not os.path.exists(METADATA_FILE):
        st.error("File metadata tidak ditemukan.")
        return None
    df = pd.read_csv(METADATA_FILE)
    df = df[df['path_gambar'].str.contains(r'dataset_kel_9[\\/]', na=False)]
    if len(df) < 10:
        st.warning("Dataset terlalu kecil untuk pelatihan Siamese Network.")
        return None

    image_paths = df['path_gambar'].tolist()
    labels = df['nama'].tolist()

    triplets = []
    for anchor_idx in range(len(image_paths)):
        anchor_path = image_paths[anchor_idx]
        anchor_label = labels[anchor_idx]
        positive_indices = [i for i, l in enumerate(labels) if l == anchor_label and i != anchor_idx]
        negative_indices = [i for i, l in enumerate(labels) if l != anchor_label]
        if positive_indices and negative_indices:
            pos_idx = np.random.choice(positive_indices)
            neg_idx = np.random.choice(negative_indices)
            triplets.append([anchor_path, image_paths[pos_idx], image_paths[neg_idx]])

    if len(triplets) < 10:
        st.warning("Jumlah triplet terlalu kecil untuk pelatihan.")
        return None

    triplet_images = []
    for anchor_path, pos_path, neg_path in triplets:
        anchor_img = cv2.imread(anchor_path)
        pos_img = cv2.imread(pos_path)
        neg_img = cv2.imread(neg_path)
        if anchor_img is None or pos_img is None or neg_img is None:
            continue
        faces_a = detect_faces(anchor_img)
        faces_p = detect_faces(pos_img)
        faces_n = detect_faces(neg_img)
        if len(faces_a) > 0 and len(faces_p) > 0 and len(faces_n) > 0:
            (xa, ya, wa, ha) = faces_a[0]
            (xp, yp, wp, hp) = faces_p[0]
            (xn, yn, wn, hn) = faces_n[0]
            face_a = preprocess_image_for_siamese(anchor_img[ya:ya+ha, xa:xa+wa])
            face_p = preprocess_image_for_siamese(pos_img[yp:yp+hp, xp:xp+wp])
            face_n = preprocess_image_for_siamese(neg_img[yn:yn+hn, xn:xn+wn])
            if face_a is not None and face_p is not None and face_n is not None:
                triplet_images.append([face_a, face_p, face_n])

    if len(triplet_images) < 10:
        st.warning("Jumlah triplet wajah yang valid terlalu kecil untuk pelatihan.")
        return None

    triplet_images = np.array(triplet_images)
    dummy_labels = np.zeros(len(triplet_images))

    siamese_model, base_network = create_siamese_model()
    siamese_model.compile(loss=triplet_loss, optimizer=Adam(0.0001))

    with st.spinner("Melatih Siamese Network..."):
        siamese_model.fit(
            [triplet_images[:, 0], triplet_images[:, 1], triplet_images[:, 2]],
            dummy_labels,
            batch_size=16,
            epochs=10,
            validation_split=0.2,
            verbose=1
        )

    siamese_model.save(SIAMESE_MODEL_PATH)
    st.success(f"Siamese model dilatih dan disimpan di {SIAMESE_MODEL_PATH}")
    return base_network

# Fungsi menghitung embedding dengan perbaikan
def compute_all_embeddings(method='facenet', siamese_model=None):
    embeddings = []
    labels = []
    paths = []
    if not os.path.exists(METADATA_FILE):
        logging.warning("File metadata tidak ditemukan.")
        return np.array(embeddings), labels, paths

    df = pd.read_csv(METADATA_FILE)
    for _, row in df.iterrows():
        img_path = row['path_gambar']
        nama = row['nama']
        image = cv2.imread(img_path)
        if image is None:
            logging.warning(f"Gambar tidak ditemukan: {img_path}")
            continue

        faces = detect_faces(image)
        if len(faces) == 0:
            logging.warning(f"Wajah tidak terdeteksi pada gambar: {img_path}")
            continue

        (x, y, w, h) = faces[0]
        face = image[y:y+h, x:x+w]

        # Inisialisasi embedding sebagai None
        embedding = None
        if method == 'facenet' and FACENET_AVAILABLE:
            embedding = get_face_embedding_facenet(face)
        elif method == 'siamese' and siamese_model is not None:
            embedding = get_face_embedding_siamese(face, siamese_model)
        elif method == 'arcface' and ARCFACE_AVAILABLE:
            embedding = get_face_embedding_arcface(face)
        else:
            logging.warning(f"Metode {method} tidak tersedia atau model Siamese belum dilatih.")
            continue

        if embedding is not None:
            embeddings.append(embedding)
            labels.append(nama)
            paths.append(img_path)
        else:
            logging.warning(f"Gagal menghitung embedding untuk gambar: {img_path}")

    return np.array(embeddings), labels, paths

# Visualisasi t-SNE
def visualize_tsne(embeddings, labels, title="t-SNE Visualization"):
    if len(embeddings) < 2:
        return None
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    unique_labels = list(set(labels))
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(unique_labels)))
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(unique_labels):
        idx = [j for j, l in enumerate(labels) if l == label]
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], c=[colors[i]], label=label, alpha=0.6)
    plt.legend()
    plt.title(title)
    return plt

# Fungsi evaluasi Face Similarity
def compute_roc_curve(embeddings, labels):
    pairs = []
    true_labels = []
    scores = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            score = 1 - cosine(embeddings[i], embeddings[j])
            pairs.append((i, j))
            scores.append(score)
            true_labels.append(1 if labels[i] == labels[j] else 0)
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return fpr, tpr, roc_auc, optimal_threshold

def compute_similarity_metrics(embeddings, labels, threshold):
    scores = []
    true_labels = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            score = 1 - cosine(embeddings[i], embeddings[j])
            scores.append(score)
            true_labels.append(1 if labels[i] == labels[j] else 0)
    
    predictions = [1 if score >= threshold else 0 for score in scores]
    cm = confusion_matrix(true_labels, predictions)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    tar = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        'TAR': tar,
        'FAR': far,
        'FRR': frr,
        'Confusion Matrix': cm
    }

# CSS Kustom dengan penyesuaian tinggi dropdown yang diperbarui
st.markdown("""
    <style>
    .stApp {
        background-color: #E6F0FA;
        color: #4A4E69;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #FF6F61;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        font-size: 16px;
        transition: background-color 0.3s;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #FF8780;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stTextInput>div>input {
        border: 2px solid #89A8C7;
        border-radius: 8px;
        background-color: #FCE4EC;
        color: #4A4E69;
        padding: 10px;
        font-size: 16px;
    }
    .stSelectbox>div>div {
        border: 2px solid #89A8C7;
        border-radius: 8px;
        background-color: #FCE4EC;
        color: #4A4E69;
        padding: 5px;
        font-size: 16px;
    }
    /* Penyesuaian tinggi dan lebar dropdown */
    .stSelectbox div[role="listbox"] {
        max-height: 500px !important; /* Tinggi maksimum dropdown ditingkatkan */
        min-width: 300px !important; /* Lebar minimum untuk menampung teks panjang */
        overflow-y: auto; /* Tambahkan scroll jika opsi terlalu banyak */
        background-color: #FCE4EC;
        border: 1px solid #89A8C7;
        border-radius: 8px;
    }
    .stSelectbox div[role="option"] {
        padding: 5px !important; /* Kurangi padding agar lebih banyak opsi yang muat */
        color: #4A4E69;
        font-size: 14px !important; /* Kurangi ukuran font untuk efisiensi ruang */
        white-space: nowrap; /* Pastikan teks tidak membungkus */
        overflow: hidden;
        text-overflow: ellipsis;
    }
    h1, h2, h3 {
        color: #4A4E69;
        font-weight: 600;
    }
    .stTabs button {
        background-color: #FCE4EC;
        color: #4A4E69;
        border: none;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-size: 16px;
        margin-right: 5px;
        transition: background-color 0.3s;
    }
    .stTabs button:hover {
        background-color: #FF8780;
        color: white;
    }
    table {
        width: 100%;
        min-width: 600px;
        border-collapse: collapse;
        font-size: 16px;
        color: #4A4E69;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    th {
        background-color: #FCE4EC;
        color: #4A4E69;
        padding: 12px;
        border: 1px solid #89A8C7;
        text-align: left;
    }
    td {
        background-color: #FFFFFF;
        color: #4A4E69;
        padding: 12px;
        border: 1px solid #89A8C7;
    }
    .stImage img {
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .stContainer {
        padding: 20px;
        border-radius: 10px;
        background-color: #FFFFFF;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    .stSidebar .stRadio > div {
        background-color: #FCE4EC;
        border-radius: 8px;
        padding: 10px;
    }
    .stSidebar .stRadio > div label {
        color: #4A4E69;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Judul
st.title("📸 Sistem Pengenalan Wajah dan Deteksi Suku")

# Inisialisasi file CSV
init_metadata_csv()

# Sidebar navigasi
st.sidebar.title("Navigasi Utama")
page = st.sidebar.radio("Pilih bagian:", [
    "Tambah Subjek",
    "Split Dataset",
    "Preprocessing",
    "Statistik",
    "Face Similarity",
    "Deteksi Suku/Etnis",
    "Deteksi Gender",
    "Evaluasi Sistem"
])

# Definisi kelas dan fungsi di tingkat global
class EthnicDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.failed_images = []  # Untuk melacak gambar yang gagal

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = cv2.imread(self.image_paths[idx])
            if image is None:
                raise ValueError(f"Gagal membaca gambar: {self.image_paths[idx]}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            label = torch.tensor(label, dtype=torch.long)
            return image, label
        except Exception as e:
            logging.error(f"Error loading image {self.image_paths[idx]}: {e}")
            self.failed_images.append((self.image_paths[idx], str(e)))
            return torch.zeros((3, 224, 224)), torch.tensor(-1, dtype=torch.long)

    def get_failed_images(self):
        return self.failed_images

def align_face(image):
    if not MTCNN_AVAILABLE:
        return cv2.resize(image, (224, 224))
    try:
        detector = MTCNN()
        results = detector.detect_faces(image)
        if not results:
            return cv2.resize(image, (224, 224))
        face = results[0]
        left_eye = face['keypoints']['left_eye']
        right_eye = face['keypoints']['right_eye']
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        faces = detect_faces(aligned)
        if faces:
            (x, y, w, h) = faces[0]
            aligned = aligned[y:y+h, x:x+w]
        return cv2.resize(aligned, (224, 224))
    except Exception as e:
        logging.error(f"Error aligning face: {e}")
        return cv2.resize(image, (224, 224))

def preprocess_for_classification(image):
    aligned_image = align_face(image)
    if aligned_image is None:
        aligned_image = cv2.resize(image, (224, 224))
    aligned_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(aligned_image).unsqueeze(0)

def initialize_ethnic_model(num_classes):
    model = models.resnet50(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, num_classes)
    )
    for param in model.fc.parameters():
        param.requires_grad = True
    return model

def evaluate_model(model, data_loader, classes, device, dataset_name="Dataset"):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            valid_mask = labels != -1
            if not valid_mask.any():
                continue
            inputs = inputs[valid_mask].to(device)
            labels = labels[valid_mask].to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    if all_labels and all_preds:
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
        st.write(f"**Classification Report ({dataset_name}):**")
        st.write(pd.DataFrame(report).transpose())
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Prediksi')
        plt.ylabel('Aktual')
        plt.title(f'Confusion Matrix ({dataset_name})')
        st.pyplot(plt)
        plt.clf()
        return cm, report, all_preds, all_labels
    else:
        st.warning(f"Tidak ada prediksi yang valid untuk evaluasi pada {dataset_name}.")
        return None, None, [], []

def train_ethnic_model(model, train_loader, val_loader, classes, num_epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, labels in train_loader:
            valid_mask = labels != -1
            if not valid_mask.any():
                continue
            inputs = inputs[valid_mask].to(device)
            labels = labels[valid_mask].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        train_loss_avg = running_loss / len(train_loader) if len(train_loader) > 0 else float('inf')

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                valid_mask = labels != -1
                if not valid_mask.any():
                    continue
                inputs = inputs[valid_mask].to(device)
                labels = labels[valid_mask].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss_avg = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        val_acc = 100 * correct / total if total > 0 else 0
        st.write(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%")

        scheduler.step(val_loss_avg)
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), MODEL_PATH)

    # Evaluasi pada data validasi setelah pelatihan selesai
    evaluate_model(model, val_loader, classes, device, dataset_name="Validasi")

def visualize_probabilities(probabilities, classes):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=probabilities, y=classes)
    plt.xlabel("Probabilitas")
    plt.title("Distribusi Probabilitas Suku/Etnis")
    st.pyplot(plt)
    plt.clf()
    
# Konten
if page == "Tambah Subjek":
    st.subheader("➕ Tambah Subjek Baru")
    with st.container():
        new_person = st.text_input("Masukkan nama subjek:", placeholder="Nama Subjek")
        suku = st.text_input("Masukkan suku/etnis:", placeholder="Suku/Etnis")
        consent = st.checkbox("Subjek telah menyetujui penggunaan foto untuk keperluan akademis.")

        variations = [
            ("tersenyum", "frontal", "indoor"),
            ("serius", "frontal", "indoor"),
            ("terkejut", "miring 45° kiri", "outdoor"),
            ("tersenyum", "miring 45° kanan", "indoor"),
            ("serius", "frontal", "redup"),
            ("tersenyum", "profil", "outdoor")
        ]

        uploaded_files = []
        for i, (ekspresi, sudut, pencahayaan) in enumerate(variations, 1):
            st.markdown(f"**Foto {i}: {ekspresi.capitalize()} – {sudut} – {pencahayaan}**")
            input_method = st.radio(f"Pilih metode input untuk Foto {i}:", ("Unggah File", "Gunakan Kamera"), key=f"method_{i}")
            if input_method == "Unggah File":
                file = st.file_uploader(f"Unggah foto {i}", type=["jpg", "png"], key=f"file_{i}")
                if file:
                    uploaded_files.append((file, ekspresi, sudut, pencahayaan))
            else:
                file = st.camera_input(f"Ambil foto {i}", key=f"camera_{i}")
                if file:
                    uploaded_files.append((file, ekspresi, sudut, pencahayaan))

        if st.button("Simpan Foto"):
            if len(uploaded_files) != len(variations):
                st.warning(f"Harap unggah atau ambil semua {len(variations)} foto.")
            elif not new_person or not suku:
                st.warning("Lengkapi nama dan suku/etnis.")
            elif not consent:
                st.warning("Dapatkan persetujuan subjek.")
            else:
                if not save_consent_form(new_person, suku):
                    st.error("Gagal menyimpan formulir persetujuan.")
                else:
                    save_path = os.path.join('dataset_kel_9', new_person, suku)
                    os.makedirs(save_path, exist_ok=True)
                    for i, (file, ekspresi, sudut, pencahayaan) in enumerate(uploaded_files, 1):
                        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        faces = detect_faces(image)
                        if len(faces) > 0:
                            (x, y, w, h) = faces[0]
                            face = image[y:y+h, x:x+w]
                            face = cv2.resize(face, (224, 224))
                        else:
                            st.warning(f"Wajah tidak terdeteksi pada Foto {i}. Menggunakan gambar asli.")
                            face = cv2.resize(image, (224, 224))
                        img_name = f"img{i}.jpg"
                        img_path = os.path.join(save_path, img_name)
                        cv2.imwrite(img_path, face)
                        add_to_metadata(img_path, new_person, suku, ekspresi, sudut, pencahayaan)
                        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        st.image(face_rgb, caption=f"Foto {i}: {ekspresi.capitalize()} – {sudut} - {pencahayaan}")
                    st.success(f"{len(uploaded_files)} gambar untuk {new_person} ({suku}) disimpan.")

elif page == "Split Dataset":
    st.subheader("📂 Split Dataset")
    with st.container():
        if st.button("Split Dataset"):
            split_dataset()

elif page == "Preprocessing":
    st.subheader("🖼️ Preprocessing Dataset")
    with st.container():
        dataset_folder = 'dataset_kel_9'
        if os.path.exists(dataset_folder):
            person_list = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]
            if person_list:
                selected_person = st.selectbox("Pilih nama:", person_list, key="preprocess_person")
                person_path = os.path.join(dataset_folder, selected_person)
                suku_list = [d for d in os.listdir(person_path) if os.path.isdir(os.path.join(person_path, d))]
                if suku_list:
                    selected_suku = st.selectbox("Pilih suku:", suku_list, key="preprocess_suku")
                    suku_path = os.path.join(dataset_folder, selected_person, selected_suku)
                    image_files = [f for f in os.listdir(suku_path) if f.endswith('.jpg')]
                    if image_files:
                        selected_image = st.selectbox("Pilih gambar:", image_files, key="preprocess_image")
                        image_path = os.path.join(suku_path, selected_image)
                        original_image = cv2.imread(image_path)
                        if original_image is not None:
                            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                            st.image(original_image_rgb, caption="Gambar Original")
                            if st.button("Proses Citra"):
                                normalized_image = cv2.resize(original_image, (224, 224))
                                rotated_image = augment_rotation(normalized_image)
                                flipped_image = augment_flip(normalized_image)
                                brightcontrast_image = augment_brightcontrast(normalized_image)
                                noisy_image = augment_gaussian_noise(normalized_image)
                                normalized_image_rgb = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2RGB)
                                rotated_image_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
                                flipped_image_rgb = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB)
                                brightcontrast_image_rgb = cv2.cvtColor(brightcontrast_image, cv2.COLOR_BGR2RGB)
                                noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
                                col1, col2, col3, col4, col5 = st.columns(5)
                                with col1:
                                    st.image(normalized_image_rgb, caption="Normalized")
                                with col2:
                                    st.image(rotated_image_rgb, caption="Rotated (±15°)")
                                with col3:
                                    st.image(flipped_image_rgb, caption="Flipped")
                                with col4:
                                    st.image(brightcontrast_image_rgb, caption="Bright/Contrast")
                                with col5:
                                    st.image(noisy_image_rgb, caption="Gaussian Noise")
                                processed_path = os.path.join('processed_dataset_kel_9', selected_person, selected_suku)
                                os.makedirs(processed_path, exist_ok=True)
                                normalized_path = os.path.join(processed_path, f"normalized_{selected_image}")
                                rotated_path = os.path.join(processed_path, f"rotated_{selected_image}")
                                flipped_path = os.path.join(processed_path, f"flipped_{selected_image}")
                                brightcontrast_path = os.path.join(processed_path, f"brightcontrast_{selected_image}")
                                noisy_path = os.path.join(processed_path, f"noisy_{selected_image}")
                                cv2.imwrite(normalized_path, normalized_image)
                                cv2.imwrite(rotated_path, rotated_image)
                                cv2.imwrite(flipped_path, flipped_image)
                                cv2.imwrite(brightcontrast_path, brightcontrast_image)
                                cv2.imwrite(noisy_path, noisy_image)
                                df = pd.read_csv(METADATA_FILE)
                                matching_rows = df[df['path_gambar'] == image_path]
                                if not matching_rows.empty:
                                    original_metadata = matching_rows.iloc[0]
                                    add_to_metadata(normalized_path, selected_person, selected_suku, original_metadata['ekspresi'], original_metadata['sudut'], original_metadata['pencahayaan'])
                                    add_to_metadata(rotated_path, selected_person, selected_suku, original_metadata['ekspresi'], original_metadata['sudut'], original_metadata['pencahayaan'])
                                    add_to_metadata(flipped_path, selected_person, selected_suku, original_metadata['ekspresi'], original_metadata['sudut'], original_metadata['pencahayaan'])
                                    add_to_metadata(brightcontrast_path, selected_person, selected_suku, original_metadata['ekspresi'], original_metadata['sudut'], original_metadata['pencahayaan'])
                                    add_to_metadata(noisy_path, selected_person, selected_suku, original_metadata['ekspresi'], original_metadata['sudut'], original_metadata['pencahayaan'])
                                    st.success("Hasil preprocessing disimpan.")
                                else:
                                    st.error(f"Tidak ditemukan metadata untuk {image_path}.")
                    else:
                        st.warning("Tidak ada gambar di folder.")
                else:
                    st.warning("Tidak ada suku di folder subjek.")
            else:
                st.warning("Belum ada subjek di dataset.")
        else:
            st.warning("Folder dataset belum ada.")

elif page == "Statistik":
    st.subheader("📊 Statistik Dataset")
    with st.container():
        if os.path.exists(METADATA_FILE):
            df = pd.read_csv(METADATA_FILE)
            unique_subjects = df.groupby(['nama', 'suku']).size().reset_index().shape[0]
            unique_suku = len(df['suku'].unique())
            total_images = len(df)
            suku_distribution = df['suku'].value_counts().reset_index()
            suku_distribution.columns = ['Suku/Etnis', 'Jumlah']
            suku_distribution.index = range(1, len(suku_distribution) + 1)
            st.write("Jumlah subjek:", unique_subjects)
            st.write("Jumlah suku/etnis:", unique_suku)
            st.write("Jumlah total gambar:", total_images)
            st.write("Distribusi suku/etnis:")
            st.table(suku_distribution)
        else:
            st.warning("Belum ada metadata dataset.")

# Bagian Face Similarity (Modifikasi pada tab "Bandingkan Wajah")
elif page == "Face Similarity":
    st.subheader("👥 Face Similarity")
    tab1, tab2, tab3, tab4 = st.tabs(["Visualisasi t-SNE", "ROC Curve", "Bandingkan Wajah", "Evaluasi Metrik"])
    
    # Inisialisasi atau muat siamese_base_model
    if os.path.exists(SIAMESE_MODEL_PATH):
        siamese_model, st.session_state.siamese_base_model = create_siamese_model()
        siamese_model.load_weights(SIAMESE_MODEL_PATH)
        st.success("Siamese model dimuat.")
    else:
        st.session_state.siamese_base_model = None

    # Tab 1: Visualisasi t-SNE (tanpa perubahan)
    with tab1:
        st.markdown("**Visualisasi t-SNE Embeddings**")
        method_tab1, method_tab2, method_tab3 = st.tabs(["Siamese Network", "FaceNet", "ArcFace"])
        
        for method, method_tab in [('siamese', method_tab1), ('facenet', method_tab2), ('arcface', method_tab3)]:
            with method_tab:
                if method == 'siamese' and st.button("Latih Siamese Network", key="train_siamese"):
                    st.session_state.siamese_base_model = train_siamese_model()
                if st.button(f"Visualisasi t-SNE ({method.capitalize()})", key=f"tsne_{method}"):
                    if method == 'siamese' and st.session_state.siamese_base_model is None:
                        st.warning("Siamese model belum dilatih atau dimuat.")
                    else:
                        with st.spinner("Menghitung embedding..."):
                            embeddings, labels, _ = compute_all_embeddings(method=method, siamese_model=st.session_state.siamese_base_model)
                        if len(embeddings) > 0:
                            tsne_plot = visualize_tsne(embeddings, labels, f"t-SNE ({method.capitalize()})")
                            if tsne_plot:
                                st.pyplot(tsne_plot)
                        else:
                            st.warning("Tidak ada embedding yang dapat divisualisasikan.")

    # Tab 2: ROC Curve (tanpa perubahan)
    with tab2:
        st.markdown("**ROC Curve**")
        method_tab1, method_tab2, method_tab3 = st.tabs(["Siamese Network", "FaceNet", "ArcFace"])
        
        for method, method_tab in [('siamese', method_tab1), ('facenet', method_tab2), ('arcface', method_tab3)]:
            with method_tab:
                if st.button(f"Tampilkan ROC Curve ({method.capitalize()})", key=f"roc_{method}"):
                    if method == 'siamese' and st.session_state.siamese_base_model is None:
                        st.warning("Siamese model belum dilatih atau dimuat.")
                    else:
                        with st.spinner("Menghitung ROC curve..."):
                            embeddings, labels, _ = compute_all_embeddings(method=method, siamese_model=st.session_state.siamese_base_model)
                        if len(embeddings) > 0:
                            fpr, tpr, roc_auc, optimal_threshold = compute_roc_curve(embeddings, labels)
                            plt.figure(figsize=(8, 6))
                            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                            plt.plot([0, 1], [0, 1], 'k--')
                            plt.scatter(fpr[np.argmax(tpr - fpr)], tpr[np.argmax(tpr - fpr)], c='red', label=f'Optimal Threshold = {optimal_threshold:.2f}')
                            plt.xlabel('False Positive Rate')
                            plt.ylabel('True Positive Rate')
                            plt.title(f'ROC Curve ({method.capitalize()})')
                            plt.legend()
                            st.pyplot(plt)
                        else:
                            st.warning("Tidak ada embedding untuk ROC curve.")

    # Tab 3: Bandingkan Wajah (Modifikasi untuk menambahkan opsi kamera)
    with tab3:
        st.markdown("**Bandingkan Dua Wajah**")
        method_tab1, method_tab2, method_tab3 = st.tabs(["Siamese Network", "FaceNet", "ArcFace"])
        
        for method, method_tab in [('siamese', method_tab1), ('facenet', method_tab2), ('arcface', method_tab3)]:
            with method_tab:
                col1, col2 = st.columns(2)
                person_list = set()
                suku_dict = {}
                
                # Mengumpulkan daftar subjek dan gambar dari dataset
                for dataset in ['dataset_kel_9', 'processed_dataset_kel_9']:
                    if os.path.exists(dataset):
                        for person in os.listdir(dataset):
                            person_path = os.path.join(dataset, person)
                            if os.path.isdir(person_path):
                                person_list.add(person)
                                suku_list = [d for d in os.listdir(person_path) if os.path.isdir(os.path.join(person_path, d))]
                                if person not in suku_dict:
                                    suku_dict[person] = {}
                                for suku in suku_list:
                                    if suku not in suku_dict[person]:
                                        suku_dict[person][suku] = []
                                    suku_path = os.path.join(person_path, suku)
                                    image_files = [f for f in os.listdir(suku_path) if f.endswith('.jpg')]
                                    for img in image_files:
                                        suku_dict[person][suku].append((dataset, img))
                
                person_list = sorted(list(person_list))

                # Gambar Pertama
                with col1:
                    st.markdown("**Gambar Pertama**")
                    source_option1 = st.radio("Pilih sumber gambar pertama:", ("Dataset", "Kamera"), key=f"source1_{method}")
                    image1 = None
                    if source_option1 == "Dataset":
                        if not person_list:
                            st.warning("Tidak ada subjek di dataset.")
                        else:
                            person1 = st.selectbox("Subjek pertama:", person_list, key=f"person1_{method}")
                            suku_list1 = list(suku_dict[person1].keys())
                            suku1 = st.selectbox("Suku pertama:", suku_list1, key=f"suku1_{method}")
                            image_files1 = [(source, img) for source, img in suku_dict[person1][suku1]]
                            image_options1 = [f"{img} ({'Original' if source == 'dataset_kel_9' else 'Processed'})" for source, img in image_files1]
                            selected_image1 = st.selectbox("Gambar pertama:", image_options1, key=f"image1_{method}")
                            source1, image1_file = image_files1[image_options1.index(selected_image1)]
                            image1_path = os.path.join(source1, person1, suku1, image1_file)
                            image1 = cv2.imread(image1_path)
                            if image1 is not None:
                                image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                                st.image(image1_rgb, caption=f"Gambar 1: {person1} ({suku1})")
                            else:
                                st.error("Gagal memuat gambar pertama.")
                    else:  # Gunakan Kamera
                        camera_input1 = st.camera_input("Ambil foto untuk gambar pertama:", key=f"camera1_{method}")
                        if camera_input1:
                            file_bytes = np.asarray(bytearray(camera_input1.read()), dtype=np.uint8)
                            image1 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                            if image1 is not None:
                                image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                                st.image(image1_rgb, caption="Gambar 1: Diambil dari Kamera")
                            else:
                                st.error("Gagal memuat gambar dari kamera.")

                # Gambar Kedua
                with col2:
                    st.markdown("**Gambar Kedua**")
                    source_option2 = st.radio("Pilih sumber gambar kedua:", ("Dataset", "Kamera"), key=f"source2_{method}")
                    image2 = None
                    if source_option2 == "Dataset":
                        if not person_list:
                            st.warning("Tidak ada subjek di dataset.")
                        else:
                            person2 = st.selectbox("Subjek kedua:", person_list, key=f"person2_{method}")
                            suku_list2 = list(suku_dict[person2].keys())
                            suku2 = st.selectbox("Suku kedua:", suku_list2, key=f"suku2_{method}")
                            image_files2 = [(source, img) for source, img in suku_dict[person2][suku2]]
                            image_options2 = [f"{img} ({'Original' if source == 'dataset_kel_9' else 'Processed'})" for source, img in image_files2]
                            selected_image2 = st.selectbox("Gambar kedua:", image_options2, key=f"image2_{method}")
                            source2, image2_file = image_files2[image_options2.index(selected_image2)]
                            image2_path = os.path.join(source2, person2, suku2, image2_file)
                            image2 = cv2.imread(image2_path)
                            if image2 is not None:
                                image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                                st.image(image2_rgb, caption=f"Gambar 2: {person2} ({suku2})")
                            else:
                                st.error("Gagal memuat gambar kedua.")
                    else:  # Gunakan Kamera
                        camera_input2 = st.camera_input("Ambil foto untuk gambar kedua:", key=f"camera2_{method}")
                        if camera_input2:
                            file_bytes = np.asarray(bytearray(camera_input2.read()), dtype=np.uint8)
                            image2 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                            if image2 is not None:
                                image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                                st.image(image2_rgb, caption="Gambar 2: Diambil dari Kamera")
                            else:
                                st.error("Gagal memuat gambar dari kamera.")

                # Proses Perbandingan
                if st.button(f"Bandingkan ({method.capitalize()})", key=f"compare_{method}"):
                    if image1 is not None and image2 is not None:
                        faces1 = detect_faces(image1)
                        faces2 = detect_faces(image2)
                        if len(faces1) > 0 and len(faces2) > 0:
                            (x1, y1, w1, h1) = faces1[0]
                            (x2, y2, w2, h2) = faces2[0]
                            face1 = image1[y1:y1+h1, x1:x1+w1]
                            face2 = image2[y2:y2+h2, x2:x2+w2]
                            if method == 'facenet':
                                embedding1 = get_face_embedding_facenet(face1)
                                embedding2 = get_face_embedding_facenet(face2)
                            elif method == 'siamese':
                                if st.session_state.siamese_base_model is None:
                                    st.error("Siamese model belum dilatih.")
                                    continue
                                embedding1 = get_face_embedding_siamese(face1, st.session_state.siamese_base_model)
                                embedding2 = get_face_embedding_siamese(face2, st.session_state.siamese_base_model)
                            elif method == 'arcface':
                                embedding1 = get_face_embedding_arcface(face1)
                                embedding2 = get_face_embedding_arcface(face2)
                            if embedding1 is not None and embedding2 is not None:
                                similarity = 1 - cosine(embedding1, embedding2)
                                embeddings, labels, _ = compute_all_embeddings(method=method, siamese_model=st.session_state.siamese_base_model)
                                if len(embeddings) > 0:
                                    _, _, _, optimal_threshold = compute_roc_curve(embeddings, labels)
                                    decision = "Match" if similarity >= optimal_threshold else "Tidak Match"
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.image(cv2.cvtColor(face1, cv2.COLOR_BGR2RGB), caption="Wajah 1")
                                    with col2:
                                        st.image(cv2.cvtColor(face2, cv2.COLOR_BGR2RGB), caption="Wajah 2")
                                    st.markdown(f"**Skor Kemiripan:** {similarity:.2f}")
                                    st.markdown(f"**Keputusan:** {decision} (Threshold: {optimal_threshold:.2f})")
                                else:
                                    st.warning("Tidak ada embedding untuk threshold.")
                            else:
                                st.error("Gagal menghitung embedding.")
                        else:
                            st.error("Wajah tidak terdeteksi pada salah satu gambar.")
                    else:
                        st.warning("Pastikan kedua gambar telah dipilih atau diambil.")

    # Tab 4: Evaluasi Metrik (tanpa perubahan)
    with tab4:
        st.markdown("**Evaluasi Metrik Face Similarity**")
        method_tab1, method_tab2, method_tab3 = st.tabs(["Siamese Network", "FaceNet", "ArcFace"])
        
        for method, method_tab in [('siamese', method_tab1), ('facenet', method_tab2), ('arcface', method_tab3)]:
            with method_tab:
                if st.button(f"Evaluasi Metrik ({method.capitalize()})", key=f"eval_{method}"):
                    if method == 'siamese' and st.session_state.siamese_base_model is None:
                        st.warning("Siamese model belum dilatih.")
                    else:
                        embeddings, labels, _ = compute_all_embeddings(method=method, siamese_model=st.session_state.siamese_base_model)
                        if len(embeddings) > 0:
                            fpr, tpr, roc_auc, optimal_threshold = compute_roc_curve(embeddings, labels)
                            metrics = compute_similarity_metrics(embeddings, labels, optimal_threshold)
                            st.write(f"**AUC:** {roc_auc:.2f}")
                            st.write(f"**TAR:** {metrics['TAR']:.2f}")
                            st.write(f"**FAR:** {metrics['FAR']:.2f}")
                            st.write(f"**FRR:** {metrics['FRR']:.2f}")
                            st.write("**Confusion Matrix:**")
                            st.write(metrics['Confusion Matrix'])
                        else:
                            st.warning("Tidak ada embedding untuk evaluasi.")

elif page == "Deteksi Suku/Etnis":
    st.subheader("🧬 Deteksi Suku/Etnis")
    with st.container():
        if 'classes' not in st.session_state:
            st.session_state.classes = None
        if 'le' not in st.session_state:
            st.session_state.le = None
        if 'transform' not in st.session_state:
            st.session_state.transform = None
        if 'train_dataset' not in st.session_state:
            st.session_state.train_dataset = None
        if 'val_dataset' not in st.session_state:
            st.session_state.val_dataset = None
        if 'test_dataset' not in st.session_state:
            st.session_state.test_dataset = None

        if st.button("Latih Model Suku/Etnis"):
            train_metadata_file = os.path.join(BASE_DIR, 'train_metadata.csv')
            val_metadata_file = os.path.join(BASE_DIR, 'val_metadata.csv')
            
            if not os.path.exists(train_metadata_file) or not os.path.exists(val_metadata_file):
                st.warning("File metadata split tidak ditemukan.")
            else:
                train_df = pd.read_csv(train_metadata_file)
                val_df = pd.read_csv(val_metadata_file)
                st.session_state.classes = sorted(set(train_df['suku']).union(set(val_df['suku'])))
                st.session_state.le = LabelEncoder()
                st.session_state.le.fit(st.session_state.classes)
                
                train_paths = train_df['path_gambar'].tolist()
                train_labels = st.session_state.le.transform(train_df['suku'])
                val_paths = val_df['path_gambar'].tolist()
                val_labels = st.session_state.le.transform(val_df['suku'])
                
                st.session_state.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                st.session_state.train_dataset = EthnicDataset(train_paths, train_labels, transform=st.session_state.transform)
                st.session_state.val_dataset = EthnicDataset(val_paths, val_labels, transform=st.session_state.transform)
                
                train_loader = DataLoader(st.session_state.train_dataset, batch_size=16, shuffle=True)
                val_loader = DataLoader(st.session_state.val_dataset, batch_size=16, shuffle=False)

                train_failed = st.session_state.train_dataset.get_failed_images()
                val_failed = st.session_state.val_dataset.get_failed_images()
                if train_failed:
                    st.error("Beberapa gambar di data training gagal diproses:")
                    for path, error in train_failed[:10]:
                        st.write(f"- {path}: {error}")
                if val_failed:
                    st.error("Beberapa gambar di data validasi gagal diproses:")
                    for path, error in val_failed[:10]:
                        st.write(f"- {path}: {error}")

                if len(train_paths) < 1 or len(val_paths) < 1:
                    st.warning("Data training atau validasi kosong. Tambahkan lebih banyak data.")
                else:
                    model = initialize_ethnic_model(len(st.session_state.classes))
                    train_ethnic_model(model, train_loader, val_loader, st.session_state.classes)
                    st.success(f"Model disimpan di {MODEL_PATH}")

        st.write("Pilih sumber gambar untuk deteksi suku:")
        option = st.radio("Sumber gambar:", ("Pilih dari dataset", "Unggah gambar", "Gunakan kamera"))
        
        image = None
        if option == "Pilih dari dataset":
            person_list = set()
            suku_dict = {}
            for dataset in ['dataset_kel_9', 'processed_dataset_kel_9']:
                if os.path.exists(dataset):
                    for person in os.listdir(dataset):
                        person_path = os.path.join(dataset, person)
                        if os.path.isdir(person_path):
                            person_list.add(person)
                            suku_list = [d for d in os.listdir(person_path) if os.path.isdir(os.path.join(person_path, d))]
                            if person not in suku_dict:
                                suku_dict[person] = {}
                            for suku in suku_list:
                                if suku not in suku_dict[person]:
                                    suku_dict[person][suku] = []
                                suku_path = os.path.join(person_path, suku)
                                image_files = [f for f in os.listdir(suku_path) if f.endswith('.jpg')]
                                for img in image_files:
                                    suku_dict[person][suku].append((dataset, img))
            
            person_list = sorted(list(person_list))
            if person_list:
                person = st.selectbox("Subjek:", person_list)
                suku_list = list(suku_dict[person].keys())
                if suku_list:
                    suku = st.selectbox("Suku:", suku_list)
                    image_files = [(source, img) for source, img in suku_dict[person][suku]]
                    image_options = [f"{img} ({'Original' if source == 'dataset_kel_9' else 'Processed'})" for source, img in image_files]
                    selected_image = st.selectbox("Gambar:", image_options)
                    source, image_file = image_files[image_options.index(selected_image)]
                    image_path = os.path.join(source, person, suku, image_file)
                    image = cv2.imread(image_path)
                    if image is not None:
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        st.image(image_rgb, caption=f"Gambar: {person} ({suku})")
                    else:
                        st.error("Gagal memuat gambar.")
            else:
                st.warning("Tidak ada subjek ditemukan di dataset.")
        elif option == "Unggah gambar":
            uploaded_file = st.file_uploader("Unggah gambar:", type=["jpg", "png"])
            if uploaded_file:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if image is not None:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    st.image(image_rgb, caption="Gambar Diunggah")
        else:  # Gunakan Kamera
            camera_input = st.camera_input("Ambil foto untuk deteksi suku:", key="camera_detection")
            if camera_input:
                file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if image is not None:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    st.image(image_rgb, caption="Gambar Diambil dari Kamera")
                else:
                    st.error("Gagal memuat gambar dari kamera.")

        if st.button("Deteksi Suku/Etnis"):
            if image is not None and os.path.exists(METADATA_FILE):
                df = pd.read_csv(METADATA_FILE)
                classes = sorted(df['suku'].unique())
                le = LabelEncoder()
                le.fit(classes)
                model = initialize_ethnic_model(len(classes))
                if os.path.exists(MODEL_PATH):
                    model.load_state_dict(torch.load(MODEL_PATH))
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = model.to(device)
                    model.eval()
                    image_tensor = preprocess_for_classification(image).to(device)
                    with torch.no_grad():
                        outputs = model(image_tensor)
                        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                        predicted_idx = np.argmax(probabilities)
                        predicted_suku = classes[predicted_idx]
                        confidence = probabilities[predicted_idx]
                    st.write(f"**Prediksi Suku:** {predicted_suku}")
                    st.write(f"**Confidence:** {confidence:.4f}")
                    visualize_probabilities(probabilities, classes)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    st.image(image_rgb, caption=f"Prediksi: {predicted_suku} ({confidence:.2%})")
                else:
                    st.warning("Model belum dilatih.")
            else:
                st.warning("Pilih, unggah, atau ambil gambar menggunakan kamera.")


# Halaman Deteksi Gender
elif page == "Deteksi Gender":
    st.subheader("🚻 Deteksi Gender")
    with st.container():
        st.write("Pilih sumber gambar untuk deteksi gender:")
        option = st.radio("Sumber gambar:", ("Pilih dari dataset", "Unggah gambar", "Gunakan kamera"), key="gender_source")
        
        image = None
        image_path = None
        if option == "Pilih dari dataset":
            dataset_dirs = ['dataset_kel_9', 'processed_dataset_kel_9']
            available_datasets = [d for d in dataset_dirs if os.path.exists(d)]
            if not available_datasets:
                st.error("Direktori dataset_kel_9 atau processed_dataset_kel_9 tidak ditemukan. Pastikan direktori ada di folder proyek.")
            else:
                person_list = set()
                suku_dict = {}
                for dataset in available_datasets:
                    for person in os.listdir(dataset):
                        person_path = os.path.join(dataset, person)
                        if os.path.isdir(person_path):
                            person_list.add(person)
                            suku_list = [d for d in os.listdir(person_path) if os.path.isdir(os.path.join(person_path, d))]
                            if person not in suku_dict:
                                suku_dict[person] = {}
                            for suku in suku_list:
                                if suku not in suku_dict[person]:
                                    suku_dict[person][suku] = []
                                suku_path = os.path.join(person_path, suku)
                                image_files = [f for f in os.listdir(suku_path) if f.lower().endswith(('.jpg', '.png'))]
                                for img in image_files:
                                    suku_dict[person][suku].append((dataset, img))
                
                person_list = sorted(list(person_list))
                if person_list:
                    person = st.selectbox("Subjek:", person_list, key="gender_person")
                    suku_list = sorted(list(suku_dict[person].keys()))
                    if suku_list:
                        suku = st.selectbox("Suku:", suku_list, key="gender_suku")
                        image_files = [(source, img) for source, img in suku_dict[person][suku]]
                        image_options = [f"{img} ({'Original' if source == 'dataset_kel_9' else 'Processed'})" for source, img in image_files]
                        selected_image = st.selectbox("Gambar:", image_options, key="gender_image")
                        source, image_file = image_files[image_options.index(selected_image)]
                        image_path = os.path.join(source, person, suku, image_file)
                        image = cv2.imread(image_path)
                        if image is not None:
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            st.image(image_rgb, caption=f"Gambar: {person} ({suku})")
                        else:
                            st.error(f"Gagal memuat gambar: {image_path}. Pastikan file ada dan formatnya didukung (JPG/PNG).")
                    else:
                        st.warning(f"Tidak ada suku ditemukan untuk subjek {person}.")
                else:
                    st.warning("Tidak ada subjek ditemukan di dataset. Pastikan struktur folder: dataset_kel_9/nama/suku/gambar.jpg")
        elif option == "Unggah gambar":
            uploaded_file = st.file_uploader("Unggah gambar:", type=["jpg", "png"], key="gender_upload")
            if uploaded_file:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if image is not None:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    st.image(image_rgb, caption="Gambar Diunggah")
                    # Simpan gambar sementara untuk DeepFace
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        cv2.imwrite(tmp_file.name, image)
                        image_path = tmp_file.name
                else:
                    st.error("Gagal memuat gambar. Pastikan file adalah JPG atau PNG yang valid.")
        else:  # Gunakan kamera
            camera_input = st.camera_input("Ambil foto untuk deteksi gender:", key="gender_camera")
            if camera_input:
                file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if image is not None:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    st.image(image_rgb, caption="Gambar Diambil dari Kamera")
                    # Simpan gambar sementara untuk DeepFace
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        cv2.imwrite(tmp_file.name, image)
                        image_path = tmp_file.name
                else:
                    st.error("Gagal memuat gambar dari kamera. Pastikan kamera berfungsi dan format gambar valid.")

        # Tombol untuk deteksi gender
        if st.button("Deteksi Gender", key="detect_gender"):
            if image is None or image_path is None:
                st.warning("Pilih, unggah, atau ambil gambar menggunakan kamera terlebih dahulu.")
            else:
                try:
                    with st.spinner("Mendeteksi gender..."):
                        # Gunakan DeepFace dengan detektor MTCNN
                        result = DeepFace.analyze(
                            img_path=image_path,
                            actions=['gender'],
                            detector_backend='mtcnn',
                            enforce_detection=True
                        )
                        predicted_gender = result[0]['dominant_gender']
                        gender_scores = result[0]['gender']
                        probabilities = [gender_scores['Man'] / 100, gender_scores['Woman'] / 100]
                        confidence = max(probabilities)
                        
                        # Tampilkan hasil
                        st.write(f"**Prediksi Gender:** {predicted_gender}")
                        st.write(f"**Confidence:** {confidence:.4f}")
                        visualize_gender_probabilities(probabilities, classes=['Male', 'Female'])
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        st.image(image_rgb, caption=f"Prediksi: {predicted_gender} ({confidence:.2%})")
                        
                        # Hapus file sementara untuk input unggah atau kamera
                        if option != "Pilih dari dataset":
                            os.remove(image_path)
                except Exception as e:
                    st.error(f"Gagal mendeteksi gender: {e}")
                    st.info("Tips: Pastikan gambar memiliki wajah yang jelas, pencahayaan cukup, dan wajah tidak terlalu miring. Coba gunakan gambar lain atau sesuaikan posisi kamera.")

elif page == "Evaluasi Sistem":
    st.subheader("📈 Evaluasi Sistem")
    with st.container():
        st.markdown("**Evaluasi Performansi Model**")
        tab1, tab2 = st.tabs(["Evaluasi Deteksi Suku/Etnis", "Evaluasi Lain (Jika Ada)"])
        
        with tab1:
            st.markdown("**Evaluasi Model Deteksi Suku/Etnis pada Data Test**")
            if st.button("Evaluasi Model Suku/Etnis pada Data Test"):
                test_metadata_file = os.path.join(BASE_DIR, 'test_metadata.csv')
                if not os.path.exists(test_metadata_file):
                    st.warning("File metadata test tidak ditemukan. Pastikan dataset sudah di-split.")
                elif not os.path.exists(MODEL_PATH):
                    st.warning("Model belum dilatih. Silakan latih model terlebih dahulu.")
                elif not (hasattr(st.session_state, 'classes') and hasattr(st.session_state, 'le') and hasattr(st.session_state, 'transform')):
                    st.warning("Silakan latih model terlebih dahulu di bagian 'Deteksi Suku/Etnis' untuk menginisialisasi data.")
                else:
                    test_df = pd.read_csv(test_metadata_file)
                    if test_df.empty:
                        st.warning("Dataset test kosong.")
                    else:
                        test_paths = test_df['path_gambar'].tolist()
                        test_labels = st.session_state.le.transform(test_df['suku'])
                        
                        if len(test_paths) < 1 or len(test_labels) < 1:
                            st.error("Data test kosong atau tidak valid. Periksa file test_metadata.csv.")
                        else:
                            st.session_state.test_dataset = EthnicDataset(test_paths, test_labels, transform=st.session_state.transform)
                            test_loader = DataLoader(st.session_state.test_dataset, batch_size=16, shuffle=False)

                            # Tampilkan gambar yang gagal diproses
                            test_failed = st.session_state.test_dataset.get_failed_images()
                            if test_failed:
                                st.error("Beberapa gambar di data test gagal diproses:")
                                for path, error in test_failed[:10]:
                                    st.write(f"- {path}: {error}")

                            model = initialize_ethnic_model(len(st.session_state.classes))
                            model.load_state_dict(torch.load(MODEL_PATH))
                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            model = model.to(device)

                            evaluate_model(model, test_loader, st.session_state.classes, device, dataset_name="Testing")

        with tab2:
            st.markdown("**Evaluasi Lain**")
            st.write("Evaluasi tambahan untuk membandingkan performa Face Similarity dan waktu komputasi.")

            # Sub-tab untuk evaluasi berbeda
            eval_tab1, eval_tab2 = st.tabs(["Perbandingan Performa Face Similarity", "Waktu Komputasi"])

            # Tab 1: Perbandingan Performa Face Similarity
            with eval_tab1:
                st.markdown("**Perbandingan Performa Siamese, FaceNet, dan ArcFace**")
                st.write("Evaluasi ini membandingkan AUC, TAR, FAR, dan FRR dari tiga metode Face Similarity berdasarkan data di `dataset_metadata.csv`.")

                if st.button("Jalankan Evaluasi Perbandingan", key="compare_similarity"):
                    if not os.path.exists(METADATA_FILE):
                        st.warning("File metadata tidak ditemukan.")
                    else:
                        methods = ['siamese', 'facenet', 'arcface']
                        results = {}
                        siamese_available = st.session_state.siamese_base_model is not None

                        for method in methods:
                            if method == 'siamese' and not siamese_available:
                                st.warning("Siamese model belum dilatih atau dimuat. Melewati evaluasi Siamese.")
                                continue
                            if method == 'facenet' and not FACENET_AVAILABLE:
                                st.warning("FaceNet tidak tersedia. Melewati evaluasi FaceNet.")
                                continue
                            if method == 'arcface' and not ARCFACE_AVAILABLE:
                                st.warning("ArcFace tidak tersedia. Melewati evaluasi ArcFace.")
                                continue

                            with st.spinner(f"Menghitung metrik untuk {method.capitalize()}..."):
                                embeddings, labels, _ = compute_all_embeddings(method=method, siamese_model=st.session_state.siamese_base_model)
                                if len(embeddings) > 0:
                                    fpr, tpr, roc_auc, optimal_threshold = compute_roc_curve(embeddings, labels)
                                    metrics = compute_similarity_metrics(embeddings, labels, optimal_threshold)
                                    results[method] = {
                                        'AUC': roc_auc,
                                        'TAR': metrics['TAR'],
                                        'FAR': metrics['FAR'],
                                        'FRR': metrics['FRR'],
                                        'Confusion Matrix': metrics['Confusion Matrix']
                                    }
                                else:
                                    st.warning(f"Tidak ada embedding untuk {method.capitalize()}.")

                        if results:
                            # Tampilkan tabel perbandingan
                            st.markdown("**Hasil Perbandingan**")
                            comparison_df = pd.DataFrame({
                                'Metode': [method.capitalize() for method in results.keys()],
                                'AUC': [results[method]['AUC'] for method in results.keys()],
                                'TAR': [results[method]['TAR'] for method in results.keys()],
                                'FAR': [results[method]['FAR'] for method in results.keys()],
                                'FRR': [results[method]['FRR'] for method in results.keys()]
                            })
                            st.table(comparison_df)

                            # Plot ROC Curve untuk semua metode
                            plt.figure(figsize=(10, 8))
                            for method in results:
                                embeddings, labels, _ = compute_all_embeddings(method=method, siamese_model=st.session_state.siamese_base_model)
                                if len(embeddings) > 0:
                                    fpr, tpr, roc_auc, _ = compute_roc_curve(embeddings, labels)
                                    plt.plot(fpr, tpr, label=f'{method.capitalize()} (AUC = {roc_auc:.2f})')
                            plt.plot([0, 1], [0, 1], 'k--')
                            plt.xlabel('False Positive Rate')
                            plt.ylabel('True Positive Rate')
                            plt.title('Perbandingan ROC Curve Face Similarity')
                            plt.legend()
                            st.pyplot(plt)
                            plt.clf()

                            # Tampilkan Confusion Matrix untuk setiap metode
                            for method in results:
                                st.markdown(f"**Confusion Matrix ({method.capitalize()})**")
                                cm = results[method]['Confusion Matrix']
                                plt.figure(figsize=(6, 4))
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
                                plt.xlabel('Prediksi')
                                plt.ylabel('Aktual')
                                plt.title(f'Confusion Matrix ({method.capitalize()})')
                                st.pyplot(plt)
                                plt.clf()
                        else:
                            st.warning("Tidak ada hasil evaluasi yang tersedia.")

            # Tab 2: Evaluasi Waktu Komputasi
            with eval_tab2:
                st.markdown("**Evaluasi Waktu Komputasi**")
                st.write("Mengukur waktu rata-rata untuk deteksi wajah, perhitungan embedding, dan prediksi suku/etnis pada 10 gambar acak.")

                if st.button("Ukur Waktu Komputasi", key="compute_time"):
                    if not os.path.exists(METADATA_FILE):
                        st.warning("File metadata tidak ditemukan.")
                    else:
                        df = pd.read_csv(METADATA_FILE)
                        if len(df) < 10:
                            st.warning("Dataset terlalu kecil (<10 gambar) untuk evaluasi waktu.")
                        else:
                            sample_paths = df['path_gambar'].sample(n=10, random_state=42).tolist()
                            methods = ['siamese', 'facenet', 'arcface']
                            timings = {'Deteksi Wajah': [], 'Embedding': [], 'Prediksi Suku': []}

                            # Inisialisasi model suku
                            classes = sorted(df['suku'].unique())
                            le = LabelEncoder()
                            le.fit(classes)
                            ethnic_model = initialize_ethnic_model(len(classes))
                            if os.path.exists(MODEL_PATH):
                                ethnic_model.load_state_dict(torch.load(MODEL_PATH))
                                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                ethnic_model = ethnic_model.to(device)
                                ethnic_model.eval()
                            else:
                                st.warning("Model suku belum dilatih. Melewati pengukuran prediksi suku.")
                                timings.pop('Prediksi Suku')

                            for method in methods:
                                if method == 'siamese' and st.session_state.siamese_base_model is None:
                                    continue
                                if method == 'facenet' and not FACENET_AVAILABLE:
                                    continue
                                if method == 'arcface' and not ARCFACE_AVAILABLE:
                                    continue

                                face_detection_times = []
                                embedding_times = []
                                prediction_times = []

                                for img_path in sample_paths:
                                    image = cv2.imread(img_path)
                                    if image is None:
                                        continue

                                    # Ukur waktu deteksi wajah
                                    start_time = time.time()
                                    faces = detect_faces(image)
                                    face_detection_times.append(time.time() - start_time)

                                    if len(faces) == 0:
                                        continue
                                    (x, y, w, h) = faces[0]
                                    face = image[y:y+h, x:x+w]

                                    # Ukur waktu embedding
                                    start_time = time.time()
                                    if method == 'facenet':
                                        embedding = get_face_embedding_facenet(face)
                                    elif method == 'siamese':
                                        embedding = get_face_embedding_siamese(face, st.session_state.siamese_base_model)
                                    elif method == 'arcface':
                                        embedding = get_face_embedding_arcface(face)
                                    embedding_times.append(time.time() - start_time)

                                    # Ukur waktu prediksi suku (jika model tersedia)
                                    if 'Prediksi Suku' in timings:
                                        start_time = time.time()
                                        image_tensor = preprocess_for_classification(image).to(device)
                                        with torch.no_grad():
                                            ethnic_model(image_tensor)
                                        prediction_times.append(time.time() - start_time)

                                # Simpan waktu rata-rata
                                timings['Deteksi Wajah'].append((method.capitalize(), np.mean(face_detection_times) if face_detection_times else 0))
                                timings['Embedding'].append((method.capitalize(), np.mean(embedding_times) if embedding_times else 0))
                                if 'Prediksi Suku' in timings and prediction_times:
                                    timings['Prediksi Suku'].append((method.capitalize(), np.mean(prediction_times)))

                            # Tampilkan hasil waktu
                            for task, times in timings.items():
                                if times:
                                    st.markdown(f"**Waktu Rata-rata {task} (detik)**")
                                    time_df = pd.DataFrame(times, columns=['Metode', 'Waktu (s)'])
                                    time_df['Waktu (s)'] = time_df['Waktu (s)'].round(4)
                                    st.table(time_df)
