"""
BBOX 色分け可視化モジュール

差分の性質（構造・傷・ナット・その他）ごとに色分けして BBOX を描画する。

色定義:
  - 構造（歪み）: 青 (255, 128, 0) BGR
  - 傷: 赤 (0, 0, 255) BGR
  - ナット: 緑 (0, 200, 0) BGR
  - その他: シアン (255, 255, 0) BGR
"""
import cv2
import numpy as np

# 色定義 (BGR)
COLOR_STRUCTURE = (255, 128, 0)   # 青系（構造・歪み）
COLOR_SCRATCH = (0, 0, 255)       # 赤（傷）
COLOR_NUT = (0, 200, 0)           # 緑（ナット）
COLOR_OTHER = (255, 255, 0)       # シアン（その他）

# ラベル
LABEL_STRUCTURE = "Structure"
LABEL_SCRATCH = "Scratch"
LABEL_NUT = "Nut"
LABEL_OTHER = "Other"


def draw_classified_bboxes(
    image,
    structure_bboxes=None,
    scratch_bboxes=None,
    nut_bboxes=None,
    other_bboxes=None,
    line_thickness=2,
    font_scale=0.5,
    draw_labels=True,
    draw_legend=True,
):
    """
    性質別に色分けした BBOX を描画する。

    Args:
        image: 描画先の画像 (BGR, カラー)
        structure_bboxes: 構造差分の BBOX リスト [(x, y, w, h), ...]
        scratch_bboxes: 傷差分の BBOX リスト
        nut_bboxes: ナット差分の BBOX リスト
        other_bboxes: その他差分の BBOX リスト
        line_thickness: 矩形の線幅
        font_scale: ラベルのフォントサイズ
        draw_labels: 各BBOXにラベルを描画するか
        draw_legend: 凡例を描画するか

    Returns:
        描画済み画像 (BGR)
    """
    result = image.copy()
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    categories = [
        (structure_bboxes or [], COLOR_STRUCTURE, LABEL_STRUCTURE),
        (scratch_bboxes or [], COLOR_SCRATCH, LABEL_SCRATCH),
        (nut_bboxes or [], COLOR_NUT, LABEL_NUT),
        (other_bboxes or [], COLOR_OTHER, LABEL_OTHER),
    ]

    for bboxes, color, label in categories:
        for i, (x, y, w, h) in enumerate(bboxes):
            # 矩形描画
            cv2.rectangle(result, (x, y), (x + w, y + h), color, line_thickness)

            if draw_labels:
                # ラベル描画（矩形の上部に背景付きテキスト）
                text = f"{label}"
                text_size, _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                text_w, text_h = text_size

                # テキスト背景
                tx = x
                ty = max(y - 4, text_h + 4)
                cv2.rectangle(
                    result,
                    (tx, ty - text_h - 4),
                    (tx + text_w + 4, ty + 2),
                    color,
                    -1,
                )
                cv2.putText(
                    result,
                    text,
                    (tx + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    if draw_legend:
        result = _draw_legend(result, categories)

    return result


def draw_zone_overlay(image, structure_zone, internal_zone, flat_zone=None, alpha=0.3):
    """
    構造領域・内部領域・平坦面を半透明で重ね描画する。

    Args:
        image: 元画像 (BGR)
        structure_zone: 構造領域マスク (0/255)
        internal_zone: 内部領域マスク (0/255)
        flat_zone: 平坦面マスク (0/255)
        alpha: 透明度

    Returns:
        描画済み画像 (BGR)
    """
    result = image.copy()
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    overlay = result.copy()

    # 構造領域を青で着色
    if structure_zone is not None:
        overlay[structure_zone > 0] = (
            overlay[structure_zone > 0] * (1 - alpha)
            + np.array([255, 128, 0]) * alpha
        ).astype(np.uint8)

    # 内部領域を薄い緑で着色
    if internal_zone is not None:
        overlay[internal_zone > 0] = (
            overlay[internal_zone > 0] * (1 - alpha / 2)
            + np.array([0, 200, 0]) * (alpha / 2)
        ).astype(np.uint8)

    # 平坦面を薄い黄色で着色（内部領域の上に重ねる）
    if flat_zone is not None:
        overlay[flat_zone > 0] = (
            overlay[flat_zone > 0] * (1 - alpha / 2)
            + np.array([0, 255, 255]) * (alpha / 2)
        ).astype(np.uint8)

    return overlay


def _draw_legend(image, categories, margin=10):
    """画像右上に凡例を描画する。"""
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 22
    box_size = 14

    # 凡例のエントリを生成（BBOXがあるカテゴリのみ）
    entries = []
    for bboxes, color, label in categories:
        if len(bboxes) > 0:
            entries.append((color, f"{label} ({len(bboxes)})"))

    if not entries:
        return image

    # 凡例の背景サイズ計算
    max_text_w = 0
    for _, text in entries:
        (tw, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
        max_text_w = max(max_text_w, tw)

    legend_w = box_size + 8 + max_text_w + margin * 2
    legend_h = len(entries) * line_height + margin * 2

    # 右上に配置
    lx = w - legend_w - margin
    ly = margin

    # 半透明背景
    overlay = image.copy()
    cv2.rectangle(overlay, (lx, ly), (lx + legend_w, ly + legend_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    # 各エントリ
    for i, (color, text) in enumerate(entries):
        ey = ly + margin + i * line_height
        # カラーボックス
        cv2.rectangle(
            image,
            (lx + margin, ey),
            (lx + margin + box_size, ey + box_size),
            color,
            -1,
        )
        # テキスト
        cv2.putText(
            image,
            text,
            (lx + margin + box_size + 8, ey + box_size - 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return image
