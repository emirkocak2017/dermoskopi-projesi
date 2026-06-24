"""
xai_engine.py
--------------
Grad-CAM tabanli aciklanabilir yapay zeka (XAI) motoru.
Cilt lezyonu triyaj modeli (EfficientNet tabanli + Squeeze-Excite blok) icin.

Uyumluluk: Python 3.9, TensorFlow 2.16.2 (Keras 3 backend), opencv-python
4.9.0.80, numpy 1.26.4.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as keras_image

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("XAIEngine")


@tf.keras.utils.register_keras_serializable()
def squeeze_excite_block(inputs, ratio: int = 8):
    from tensorflow.keras import layers

    filters = inputs.shape[-1]
    x = layers.GlobalAveragePooling2D()(inputs)
    x = layers.Dense(filters // ratio, activation="relu")(x)
    x = layers.Dense(filters, activation="sigmoid")(x)
    x = layers.Reshape((1, 1, filters))(x)
    return layers.Multiply()([inputs, x])


class GradCAMError(RuntimeError):
    """Grad-CAM islem hattina ozel, anlasilir hata sinifi."""


class UltimateXAI:
    def __init__(self, model_path: str = "ultimate_skin_model.keras",
                 target_layer_name: Optional[str] = None):
        self.model_path = Path(model_path)
        self.model = self._load_model()

        owner = None
        if target_layer_name:
            name, owner = target_layer_name, self._find_owner(self.model, target_layer_name)
            if owner is None:
                raise GradCAMError(f"Belirtilen katman bulunamadi: {target_layer_name}")
        else:
            name, owner = self._find_target_layer()

        self.target_layer_name = name
        self._nested_owner = owner if owner is not self.model else None
        logger.info(
            f"Kullanilan katman: '{self.target_layer_name}'"
            + (f" (alt-model: '{owner.name}')" if self._nested_owner is not None else "")
        )
        self._remaining_layers: List = []
        self._before_model = None
        self._after_model = None
        self.grad_model = None
        self._build_grad_pipeline()

    def _load_model(self) -> tf.keras.Model:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model dosyasi bulunamadi: {self.model_path}")

        custom_objects = {"squeeze_excite_block": squeeze_excite_block}
        try:
            model = tf.keras.models.load_model(
                self.model_path, custom_objects=custom_objects, safe_mode=False
            )
        except TypeError:
            try:
                model = tf.keras.models.load_model(self.model_path, custom_objects=custom_objects)
            except Exception as e:
                raise GradCAMError(f"Model yuklenemedi (legacy yol): {e}") from e
        except (OSError, ValueError) as e:
            raise GradCAMError(f"Model dosyasi okunamadi/bozuk: {e}") from e
        except Exception as e:
            raise GradCAMError(f"Model yuklenirken beklenmeyen hata: {e}") from e

        logger.info(f"Model yuklendi: {self.model_path.name} ({len(model.layers)} katman)")
        return model

    @staticmethod
    def _safe_4d_shape(layer) -> Optional[Tuple]:
        # Keras 3 icin burası kritik dense veya conv fark etmeksizin output shape patlamasin diye hasattr yapiyoruz
        try:
            out = layer.output
        except AttributeError:
            return None
        except Exception:
            return None
        shape = getattr(out, "shape", None)
        return tuple(shape) if shape is not None else None

    @staticmethod
    def _connected_output(layer):
        # efficientnet gibi ic ice modellerde output_tensors'ın son node'unu yakalamak icin baglantiyi elle aliyoruz
        try:
            nodes = layer._inbound_nodes
            if nodes:
                outs = nodes[-1].output_tensors
                return outs[0] if isinstance(outs, (list, tuple)) and len(outs) == 1 else outs
        except (AttributeError, IndexError):
            pass
        return layer.output

    def _find_target_layer(self) -> Tuple[str, tf.keras.Model]:
        # tersten tarayarak 4 boyutlu cıkan en son konvolusyon ktamanini seciyoruz
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue  
            shape = self._safe_4d_shape(layer)
            if shape is not None and len(shape) == 4:
                return layer.name, self.model

        # eger ust katmanda 4d bulamazsak pooling falan vardır mecbur alt modele dalacaz
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.Model):
                for sub_layer in reversed(layer.layers):
                    if isinstance(sub_layer, tf.keras.layers.InputLayer):
                        continue
                    shape = self._safe_4d_shape(sub_layer)
                    if shape is not None and len(shape) == 4:
                        logger.warning(
                            f"Ust seviyede 4D katman yok; nested katman "
                            f"'{sub_layer.name}' (alt-model: '{layer.name}') "
                            f"kullaniliyor."
                        )
                        return sub_layer.name, layer

        raise GradCAMError(
            "Grad-CAM icin uygun (4 boyutlu cikis veren) bir evrisim "
            "katmani bulunamadi. target_layer_name'i manuel belirtin."
        )

    @staticmethod
    def _find_owner(model, name) -> Optional[tf.keras.Model]:
        for layer in model.layers:
            if layer.name == name:
                return model
            if isinstance(layer, tf.keras.Model):
                owner = UltimateXAI._find_owner(layer, name)
                if owner is not None:
                    return owner
        return None

    def _build_grad_pipeline(self) -> None:
        if self._nested_owner is None:
            # hedef katman en ust seviyedeyse sorun yok direkt bagla
            target_layer = self.model.get_layer(self.target_layer_name)
            target_output = self._connected_output(target_layer)
            try:
                self.grad_model = tf.keras.models.Model(
                    self.model.inputs, [target_output, self.model.output]
                )
            except Exception as e:
                raise GradCAMError(
                    f"Grad-CAM modeli kurulamadi (katman: "
                    f"'{self.target_layer_name}'): {e}"
                ) from e
            return

        # eger hedef icerdeyse (efficientnet ici gibi) modeli once ve sonra diye parcalayip zincir kurmamiz lazim
        owner = self._nested_owner
        target_layer = owner.get_layer(self.target_layer_name)
        try:
            self._before_model = tf.keras.models.Model(owner.input, target_layer.output)
            self._after_model = tf.keras.models.Model(target_layer.output, owner.output)
        except Exception as e:
            raise GradCAMError(
                f"Nested katman icin once/sonra modelleri kurulamadi: {e}"
            ) from e

        found_owner = False
        for layer in self.model.layers:
            if layer is owner:
                found_owner = True
                continue
            if not found_owner or isinstance(layer, tf.keras.layers.InputLayer):
                continue
            self._remaining_layers.append(layer)

    @staticmethod
    def _preprocess(img_path: Path, target_size=(224, 224)) -> Tuple[np.ndarray, np.ndarray]:
        img = keras_image.load_img(img_path, target_size=target_size)
        img_arr = keras_image.img_to_array(img)
        img_batch = tf.keras.applications.efficientnet.preprocess_input(
            np.expand_dims(img_arr, axis=0)
        )
        return img_arr, img_batch

    def _forward(self, img_batch: np.ndarray):
        # tek bir gradient tape icinde calisacak sekilde ileri beslemeyi hallet
        if self.grad_model is not None:
            conv_outputs, predictions = self.grad_model(img_batch)
            return conv_outputs, predictions

        conv_outputs = self._before_model(img_batch)
        x = self._after_model(conv_outputs)
        for layer in self._remaining_layers:
            x = layer(x)
        return conv_outputs, x

    def _compute_heatmap(self, img_batch: np.ndarray, class_index: Optional[int] = None):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self._forward(img_batch)
            tape.watch(conv_outputs)

            if conv_outputs.shape.rank != 4:
                raise GradCAMError(
                    f"Beklenmeyen tensor boyutu: '{self.target_layer_name}' "
                    f"katmaninin cikisi {conv_outputs.shape.rank}D "
                    f"(sekil={conv_outputs.shape}). Grad-CAM 4D "
                    f"(batch,H,W,C) gerektirir."
                )

            target_class = class_index if class_index is not None else int(
                tf.argmax(predictions[0])
            )
            loss = predictions[:, target_class]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            raise GradCAMError(
                "Gradyan hesaplanamadi (None). Hedef katman, secilen sinif "
                "cikisina hesaplama grafigi uzerinden baglanamadi."
            )

        # boyuttan bagimsiz dinamik reduction axis=(0,1,2) sabit verince cikan hatayi cozuyor
        reduce_axes = tuple(range(grads.shape.rank - 1))
        pooled_grads = tf.reduce_mean(grads, axis=reduce_axes)

        conv_outputs_single = conv_outputs[0]
        heatmap = conv_outputs_single @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)

        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        return heatmap.numpy(), target_class, predictions.numpy()[0]

    @staticmethod
    def _overlay(img_arr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5):
        h, w = img_arr.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        img_uint8 = np.uint8(img_arr)
        superimposed = cv2.addWeighted(img_uint8, 1 - alpha, heatmap_color_rgb, alpha, 0)
        return heatmap_color_rgb, superimposed

    def generate_analysis(self, img_path: str, class_index: Optional[int] = None,
                           output_dir: str = ".") -> Path:
        img_path = Path(img_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Goruntu bulunamadi: {img_path}")

        logger.info(f"Analiz ediliyor: {img_path.name}")
        img_arr, img_batch = self._preprocess(img_path)
        heatmap, pred_class, probs = self._compute_heatmap(img_batch, class_index)
        heatmap_color, superimposed = self._overlay(img_arr, heatmap)

        logger.info(f"Tahmin edilen sinif: {pred_class} (guven: {float(probs[pred_class]):.3f})")

        # plt ile 3lu dashboard cizimi yapıp diske veriyoruz
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(np.uint8(img_arr))
        axes[0].set_title("Orijinal Goruntu")
        axes[0].axis("off")

        axes[1].imshow(heatmap_color)
        axes[1].set_title("Grad-CAM Isi Haritasi")
        axes[1].axis("off")

        axes[2].imshow(superimposed)
        axes[2].set_title(f"Bindirilmis Sonuc (Sinif {pred_class})")
        axes[2].axis("off")

        fig.suptitle(f"XAI Analizi: {img_path.stem}", fontsize=14)
        fig.tight_layout()

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"analiz_{img_path.stem}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Basarili! Panel kaydedildi: {out_path}")
        return out_path


if __name__ == "__main__":
    try:
        engine = UltimateXAI(model_path="ultimate_skin_model.keras")
    except (FileNotFoundError, GradCAMError) as e:
        logger.error(f"Motor baslatilamadi: {e}")
        raise SystemExit(1)

    while True:
        path = input("\nFotograf yolu (q: cikis): ").strip()
        if path.lower() == "q":
            break
        try:
            engine.generate_analysis(path)
        except (FileNotFoundError, GradCAMError) as e:
            logger.error(str(e))
        except Exception as e:
            logger.error(f"Beklenmeyen hata: {e}")