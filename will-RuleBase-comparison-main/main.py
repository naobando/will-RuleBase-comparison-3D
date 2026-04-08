"""
板金画像比較分析ツール（HEIC対応）
Google Colab上で2枚の板金画像を比較し、差分（キズ・打痕・汚れ・位置ズレ等）を
類似度（MSE/SSIM）で定量評価し、差分ヒートマップ/SSIMマップで可視化します。
"""

# =========================================================
# 0) 必要なライブラリのインポート
# =========================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import io
from PIL import Image

# HEIC / HEIF をPILで開けるように登録
from pillow_heif import register_heif_opener

try:
    from google.colab import files
    _HAVE_COLAB = True
except Exception:
    files = None
    _HAVE_COLAB = False

register_heif_opener()

# 日本語フォント設定(Google Colab用)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def preprocess_image(image, target_size=None, blur_ksize=3):
    """
    画像の前処理を行う関数（板金向け）
    Args:
        image: 入力画像(カラーまたはグレースケール)
        target_size: リサイズ後のサイズ (width, height) のタプル。Noneの場合はリサイズしない
        blur_ksize: ガウシアンブラーのカーネルサイズ(奇数)
    Returns:
        前処理済みのグレースケール画像
    """
    # グレースケールに変換(既にグレースケールの場合はそのまま)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # リサイズ処理
    if target_size is not None:
        gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

    # ノイズ除去(ガウシアンブラー)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return blurred


def calculate_mse(imageA, imageB):
    """
    2枚の画像間のMSE(Mean Squared Error)を計算
    Args:
        imageA, imageB: 比較する2枚の画像
    Returns:
        MSE値(0に近いほど類似)
    """
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def diff_to_bboxes(
    diff_gray,
    thresh=13,
    min_area=1500,
    morph_kernel=21,
    close_iter=4,
    open_iter=1,
    max_boxes=6,
):
    """
    差分(diff)画像から「白が密集した塊」だけをBBOX化（板金向け）
    仕組み：
      1) 二値化（閾値以上を白）
      2) Openで小ノイズ除去
      3) Closeで近い白を結合＆穴埋め → “塊”にする
      4) 連結成分を面積順に並べ、上位max_boxesだけ残す
    Args:
        diff_gray: 差分画像（グレースケール, 0-255）
        thresh: 差分閾値（小さいほど拾う）
        min_area: 最小面積(px)（小さい塊を捨てる）
        morph_kernel: まとめる強さ（大きいほど塊になる）
        close_iter: 近接領域の結合＆穴埋めの強さ
        open_iter: 小ノイズ除去の強さ
        max_boxes: 表示するBBOX上限（例：5〜6）
    Returns:
        mask: 二値マスク（処理後）
        bboxes: [(x,y,w,h), ...]（面積の大きい順、最大max_boxes）
    """
    # 1) 二値化
    _, mask = cv2.threshold(diff_gray, thresh, 255, cv2.THRESH_BINARY)

    # 2) モルフォロジー（ノイズ除去→結合）
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    if open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=open_iter)
    if close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=close_iter)

    # 3) 連結成分（塊）抽出
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # stats: [label, x, y, w, h, area] ではなく (x,y,w,h,area) が入る
    bboxes = []
    for label in range(1, num_labels):  # 0は背景
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        bboxes.append((x, y, w, h, area))

    # 4) 面積の大きい順に上位だけ残す
    bboxes = sorted(bboxes, key=lambda b: b[4], reverse=True)
    bboxes = bboxes[:max_boxes]

    # (x,y,w,h) に戻す
    bboxes_xywh = [(x, y, w, h) for (x, y, w, h, area) in bboxes]
    return mask, bboxes_xywh


def compare_images(
    imageA,
    imageB,
    title="Sheet Metal Image Comparison",
    diff_thresh=13,
    min_area=1500,
    morph_kernel=20,
    max_boxes=6,
    show_plot=True,
):
    # サイズ合わせ
    if imageA.shape != imageB.shape:
        target_size = (imageA.shape[1], imageA.shape[0])
        imageB = cv2.resize(imageB, target_size, interpolation=cv2.INTER_AREA)

    # 前処理（グレー＋ブラー）
    processedA = preprocess_image(imageA, blur_ksize=3)
    processedB = preprocess_image(imageB, blur_ksize=3)

    # 指標
    mse_value = calculate_mse(processedA, processedB)
    ssim_score, ssim_map = ssim(processedA, processedB, full=True)

    # 差分
    diff = cv2.absdiff(processedA, processedB)

    # マスク→BBOX（塊だけ）
    mask, bboxes = diff_to_bboxes(
        diff_gray=diff,
        thresh=diff_thresh,
        min_area=min_area,
        morph_kernel=morph_kernel,
        close_iter=4,
        open_iter=1,
        max_boxes=max_boxes,
    )

    # BBOX描画
    visA = imageA.copy()
    visB = imageB.copy()
    for (x, y, w, h) in bboxes:
        cv2.rectangle(visA, (x, y), (x + w, y + h), (0, 0, 255), 4)
        cv2.rectangle(visB, (x, y), (x + w, y + h), (0, 0, 255), 4)

    print("=" * 60)
    print(f"【{title}】")
    print("=" * 60)
    print(f"MSE: {mse_value:.2f}（0に近いほど類似）")
    print(f"SSIM: {ssim_score:.4f}（1に近いほど類似）")
    print(f"検出BBOX数（上限{max_boxes}）: {len(bboxes)}")
    print(f"diff_thresh={diff_thresh}, min_area={min_area}, morph_kernel={morph_kernel}")
    print("=" * 60)

    # ヒートマップとSSIMも残す
    fig = plt.figure(figsize=(20, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(visA, cv2.COLOR_BGR2RGB))
    plt.title("Image A (Master) + BBOX", fontweight="bold")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(visB, cv2.COLOR_BGR2RGB))
    plt.title("Image B (Test) + BBOX", fontweight="bold")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(mask, cmap="gray")
    plt.title("Diff Mask（塊化後）", fontweight="bold")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    im = plt.imshow(diff, cmap="jet")
    plt.title("Difference Heatmap（絶対差分）", fontweight="bold")
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.subplot(2, 3, 5)
    ssim_display = (ssim_map + 1) / 2
    im2 = plt.imshow(ssim_display, cmap="plasma", vmin=0, vmax=1)
    plt.title("SSIM Map", fontweight="bold")
    plt.axis("off")
    plt.colorbar(im2, fraction=0.046, pad=0.04)

    plt.subplot(2, 3, 6)
    plt.axis("off")
    bbox_text = "\n".join(
        [f"{i+1}. x={x}, y={y}, w={w}, h={h}" for i, (x, y, w, h) in enumerate(bboxes)]
    )
    plt.text(0, 1, bbox_text if bbox_text else "No BBOX detected", va="top", fontsize=10)
    plt.title("BBOX list", fontweight="bold")
    plt.suptitle(
        f"{title}\nBBOX={len(bboxes)} | SSIM={ssim_score:.4f} | MSE={mse_value:.2f}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    if show_plot:
        plt.show()

    return mse_value, ssim_score, diff, mask, bboxes, fig


def load_image_from_bytes(file_bytes):
    pil_img = Image.open(io.BytesIO(file_bytes))
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def upload_and_load_images():
    """
    Google Colabで画像をアップロードし、読み込む関数（HEIC対応）
    Returns:
        uploaded_images: {ファイル名: 画像データ(OpenCV BGR)} の辞書
    """
    if not _HAVE_COLAB:
        raise RuntimeError("google.colab が利用できません。Colab上で実行してください。")
    print("画像ファイルを選択してアップロードしてください...")
    print("(複数ファイルを選択可能です)")
    print("対応形式: JPG / PNG / HEIC / HEIF")
    uploaded = files.upload()

    uploaded_images = {}
    for filename, file_data in uploaded.items():
        try:
            image_cv = load_image_from_bytes(file_data)
            uploaded_images[filename] = image_cv
            print(f"✓ {filename} を読み込みました (サイズ: {image_cv.shape})")
        except Exception as e:
            print(f"✗ {filename} の読み込みに失敗しました: {e}")

    return uploaded_images


# ================================
# メイン処理
# ================================
if __name__ == "__main__":
    print("【板金画像比較分析ツール（HEIC対応）】")
    print("2枚の画像をアップロードして比較分析を行います\n")

    # 画像のアップロード
    images = upload_and_load_images()

    # アップロードされた画像のリスト
    image_list = list(images.items())
    if len(image_list) < 2:
        print("\n⚠ エラー: 比較には2枚以上の画像が必要です")
    else:
        print(f"\n{len(image_list)}枚の画像がアップロードされました")
        print("\n最初の2枚の画像を比較します...")

        # 最初の2枚を比較
        filename_a, image_a = image_list[0]
        filename_b, image_b = image_list[1]
        print(f"\n比較対象:")
        print(f" Image A (Master): {filename_a}")
        print(f" Image B (Test) : {filename_b}\n")

        # 比較実行
        mse, ssim_score, diff, mask, bboxes, _ = compare_images(
            image_a,
            image_b,
            title=f"Sheet Metal Comparison: {filename_a} vs {filename_b}",
            diff_thresh=12,  # まず15
            min_area=300,  # 小さいノイズを捨てる
            morph_kernel=20,  # まとめる強さ（大きめ）
            max_boxes=6,  # 5〜6個にしたい
        )
        print("検出BBOX数:", len(bboxes))

        # 複数ペアがある場合の処理例
        if len(image_list) >= 4:
            print("\n\n他のペアも比較しますか? (3枚目と4枚目)")
            response = input("続行する場合は 'y' を入力: ")
            if response.lower() == 'y':
                filename_c, image_c = image_list[2]
                filename_d, image_d = image_list[3]
                print(f"\n比較対象:")
                print(f" Image C (Master): {filename_c}")
                print(f" Image D (Test) : {filename_d}\n")

                mse2, ssim_score2, diff2, _, _, _ = compare_images(
                    image_c,
                    image_d,
                    title=f"Sheet Metal Comparison: {filename_c} vs {filename_d}",
                )

    print("\n\n分析が完了しました!")
    print("必要に応じて、compare_images()関数を直接呼び出して")
    print("任意の画像ペアを比較できます。")

# ================================
# 使用例(手動で画像を指定する場合)
# ================================
"""
# 既に画像が読み込まれている場合の使用例:
# 例1: アップロードした画像を使用
images = upload_and_load_images()
img1 = list(images.values())[0]
img2 = list(images.values())[1]
mse, ssim_score, diff = compare_images(img1, img2, title="Manual Comparison")

# 例2: ファイルパスから直接読み込み(Colab環境の場合)
# img1 = cv2.imread('/content/image1.jpg')
# img2 = cv2.imread('/content/image2.jpg')
# mse, ssim_score, diff = compare_images(img1, img2, title="Path-based Comparison")
"""