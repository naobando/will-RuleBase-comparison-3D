import json
import os
import time
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

from config import load_config
from main import compare_images


config = load_config()

st.set_page_config(page_title="板金画像比較分析ツール", layout="wide")

st.title(config["ui"]["title"])
st.write(config["ui"]["description"])

try:
    _AUTOREFRESH = st.autorefresh
except AttributeError:
    try:
        from streamlit import st_autorefresh

        _AUTOREFRESH = st_autorefresh
    except Exception:
        _AUTOREFRESH = None


def capture_frame(device_index):
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError("カメラを開けませんでした。デバイス番号をご確認ください。")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("フレームを取得できませんでした。")
    return frame


def _resolve_guide_path(master_dir, master_image):
    if master_image:
        return os.path.join(os.getcwd(), master_dir, master_image)
    if not os.path.isdir(master_dir):
        return None
    for name in os.listdir(master_dir):
        lower = name.lower()
        if lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            return os.path.join(master_dir, name)
    return None


def _rotate_image(image, angle_deg, is_mask=False):
    if angle_deg == 0 or image is None:
        return image
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    flags = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    border_value = 0 if is_mask else (0, 0, 0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _rotate_live_frame(frame, angle_deg):
    if angle_deg == 0 or frame is None:
        return frame
    angle = angle_deg % 360
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return _rotate_image(frame, angle, is_mask=False)


def _load_guide_mask(path):
    if not path or not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        mask = img[:, :, 3]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _rotate_image(mask, config["guide"]["rotate_deg"], is_mask=True)


def _apply_guide(frame, mask, scale, alpha, color=(0, 255, 0)):
    if mask is None:
        return frame
    h, w = frame.shape[:2]
    target_w = max(1, int(w * scale))
    scale_ratio = target_w / mask.shape[1]
    target_h = max(1, int(mask.shape[0] * scale_ratio))
    if target_h > h:
        scale_ratio = h / mask.shape[0]
        target_h = h
        target_w = max(1, int(mask.shape[1] * scale_ratio))

    mask_resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    y0 = (h - target_h) // 2
    x0 = (w - target_w) // 2
    roi = frame[y0 : y0 + target_h, x0 : x0 + target_w]

    alpha_mask = (mask_resized.astype(np.float32) / 255.0) * alpha
    if alpha_mask.ndim == 2:
        alpha_mask = alpha_mask[:, :, None]
    overlay = np.zeros_like(roi, dtype=np.float32)
    overlay[:] = color
    blended = (1 - alpha_mask) * roi.astype(np.float32) + alpha_mask * overlay
    frame[y0 : y0 + target_h, x0 : x0 + target_w] = blended.astype(np.uint8)
    return frame


def _load_master_image():
    guide_path = _resolve_guide_path(
        config["guide"]["master_dir"],
        config["guide"]["master_image"],
    )
    if not guide_path or not os.path.exists(guide_path):
        return None
    image = cv2.imread(guide_path, cv2.IMREAD_COLOR)
    return _rotate_image(image, config["guide"]["rotate_deg"], is_mask=False)


def _set_live(value):
    st.session_state["live"] = value
    if not value:
        st.session_state.pop("analysis_save_error", None)
        st.session_state.pop("test_frame", None)
        st.session_state.pop("last_frame", None)
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("analysis_error", None)


def _run_analysis(diff_thresh, min_area, morph_kernel, max_boxes):
    master_frame = _load_master_image()
    test_frame = st.session_state.get("test_frame")
    if master_frame is None or test_frame is None:
        if master_frame is None:
            st.session_state["analysis_error"] = "マスター画像が見つかりません。"
        else:
            st.session_state["analysis_error"] = "比較画像を撮影してください。"
        st.session_state.pop("analysis_result", None)
        return

    try:
        result = compare_images(
            master_frame,
            test_frame,
            title="Sheet Metal Comparison: Master vs Test",
            diff_thresh=diff_thresh,
            min_area=min_area,
            morph_kernel=morph_kernel,
            max_boxes=max_boxes,
            show_plot=False,
        )
        st.session_state["analysis_result"] = result
        st.session_state["analysis_params"] = (diff_thresh, min_area, morph_kernel, max_boxes)
        st.session_state.pop("analysis_error", None)
        try:
            _save_result(
                master_frame,
                test_frame,
                result,
                diff_thresh,
                min_area,
                morph_kernel,
                max_boxes,
                base_dir=config["output"]["base_dir"],
            )
            st.session_state.pop("analysis_save_error", None)
        except Exception as exc:
            st.session_state["analysis_save_error"] = str(exc)
    except Exception as exc:
        st.session_state["analysis_error"] = str(exc)
        st.session_state.pop("analysis_result", None)


def _save_result(
    master_frame,
    test_frame,
    result,
    diff_thresh,
    min_area,
    morph_kernel,
    max_boxes,
    base_dir,
):
    mse, ssim_score, diff, mask, bboxes, fig = result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.getcwd(), base_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(os.path.join(out_dir, "master.png"), master_frame)
    cv2.imwrite(os.path.join(out_dir, "test.png"), test_frame)
    cv2.imwrite(os.path.join(out_dir, "diff.png"), diff)
    cv2.imwrite(os.path.join(out_dir, "mask.png"), mask)
    fig.savefig(os.path.join(out_dir, "result.png"), dpi=150)

    meta = {
        "timestamp": timestamp,
        "mse": float(mse),
        "ssim": float(ssim_score),
        "bboxes": bboxes,
        "params": {
            "diff_thresh": diff_thresh,
            "min_area": min_area,
            "morph_kernel": morph_kernel,
            "max_boxes": max_boxes,
        },
    }
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def _request_analysis():
    st.session_state["live"] = False
    st.session_state["analysis_request_id"] = st.session_state.get("analysis_request_id", 0) + 1


left, center, right = st.columns([1, 2, 2])

with left:
    st.subheader("パラメータ / 操作")
    st.markdown("**検出パラメータ**")
    diff_thresh = st.slider("diff_thresh", 1, 50, config["analysis"]["diff_thresh"])
    min_area = st.slider("min_area", 10, 5000, config["analysis"]["min_area"])
    morph_kernel = st.slider(
        "morph_kernel", 3, 51, config["analysis"]["morph_kernel"], step=2
    )
    max_boxes = st.slider("max_boxes", 1, 10, config["analysis"]["max_boxes"])

    st.divider()
    st.markdown("**カメラ設定**")
    camera_enabled = st.toggle("カメラ接続を有効化", value=True)
    device_index = st.number_input(
        "カメラデバイス番号",
        min_value=0,
        max_value=10,
        value=config["camera"]["device_index"],
        step=1,
    )
    st.caption("連係カメラが認識されない場合は番号を変えてお試しください。")

    st.divider()
    st.markdown("**操作**")
    start_live = st.button(
        "カメラ接続", on_click=_set_live, args=(True,), disabled=not camera_enabled
    )
    stop_live = st.button(
        "カメラ接続解除", on_click=_set_live, args=(False,), disabled=not camera_enabled
    )
    capture_test = st.button("比較画像を撮影", disabled=not camera_enabled)
    run_analysis = st.button("解析実行", on_click=_request_analysis)

with center:
    st.subheader("ライブ映像")
    if "live" not in st.session_state:
        st.session_state["live"] = False
    if "last_frame" not in st.session_state:
        st.session_state["last_frame"] = None

    try:
        if not camera_enabled:
            st.session_state["live"] = False
            st.session_state["last_frame"] = None
            st.info("カメラ接続が無効です。左で有効化してください。")
        else:
            if capture_test:
                frame = capture_frame(device_index)
                st.session_state["last_frame"] = frame

            if st.session_state["live"]:
                frame = capture_frame(device_index)
                st.session_state["last_frame"] = frame

            if st.session_state["last_frame"] is not None:
                guide_mask = st.session_state.get("guide_mask")
                if guide_mask is None and config["guide"]["enabled"]:
                    guide_path = _resolve_guide_path(
                        config["guide"]["master_dir"],
                        config["guide"]["master_image"],
                    )
                    guide_mask = _load_guide_mask(guide_path)
                    st.session_state["guide_mask"] = guide_mask
                frame_to_show = st.session_state["last_frame"].copy()
                frame_to_show = _rotate_live_frame(
                    frame_to_show, config["camera"]["live_rotate_deg"]
                )
                if config["guide"]["enabled"] and guide_mask is not None:
                    frame_to_show = _apply_guide(
                        frame_to_show,
                        guide_mask,
                        scale=config["guide"]["scale"],
                        alpha=config["guide"]["alpha"],
                    )
                st.image(
                    cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB),
                    caption="現在フレーム",
                    use_container_width=True,
                )
                if config["guide"]["enabled"] and guide_mask is None:
                    st.warning("ガイド用マスター画像が見つかりません。")
            else:
                st.info("ライブ表示を開始するか、撮影ボタンを押してください。")

            if capture_test:
                if st.session_state["last_frame"] is None:
                    st.session_state["last_frame"] = capture_frame(device_index)
                st.session_state["test_frame"] = st.session_state["last_frame"]
                st.success("比較画像を保存しました。")
                st.session_state.pop("analysis_error", None)
                st.session_state.pop("analysis_save_error", None)
    except Exception as exc:
        st.error(f"カメラ取得に失敗しました: {exc}")

    if st.session_state["live"]:
        if _AUTOREFRESH is not None:
            _AUTOREFRESH(interval=config["camera"]["live_refresh_ms"], key="live_refresh")
        else:
            time.sleep(0.2)
            st.rerun()

with right:
    st.subheader("基準 / 比較画像")
    master_frame = _load_master_image()
    test_frame = st.session_state.get("test_frame")
    image_col_left, image_col_right = st.columns(2)
    with image_col_left:
        if master_frame is not None:
            st.image(
                cv2.cvtColor(master_frame, cv2.COLOR_BGR2RGB),
                caption="基準画像",
                use_container_width=True,
            )
        else:
            st.warning("マスター画像が見つかりません。")
    with image_col_right:
        if test_frame is not None:
            st.image(
                cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB),
                caption="比較画像",
                use_container_width=True,
            )
        else:
            st.info("比較画像が未撮影です。")

    st.subheader("解析結果")
    if "analysis_error" in st.session_state:
        st.error(f"解析に失敗しました: {st.session_state['analysis_error']}")
    elif "analysis_result" in st.session_state:
        mse, ssim_score, diff, mask, bboxes, fig = st.session_state["analysis_result"]
        st.write(f"MSE: {mse:.2f}（0に近いほど類似）")
        st.write(f"SSIM: {ssim_score:.4f}（1に近いほど類似）")
        st.write(f"検出BBOX数（上限{max_boxes}）: {len(bboxes)}")
        st.pyplot(fig, clear_figure=True)
        if "analysis_save_error" in st.session_state:
            st.warning(f"保存に失敗しました: {st.session_state['analysis_save_error']}")
        if master_frame is None or test_frame is None:
            st.info("新規解析には基準/比較画像の撮影が必要です。")
    elif master_frame is None or test_frame is None:
        if master_frame is None:
            st.info("マスター画像を配置してください。")
        else:
            st.info("比較画像を撮影してください。")
    else:
        req_id = st.session_state.get("analysis_request_id", 0)
        res_id = st.session_state.get("analysis_result_id", 0)
        if req_id > res_id:
            _run_analysis(diff_thresh, min_area, morph_kernel, max_boxes)
            st.session_state["analysis_result_id"] = req_id
            if "analysis_result" in st.session_state:
                mse, ssim_score, diff, mask, bboxes, fig = st.session_state["analysis_result"]
                st.write(f"MSE: {mse:.2f}（0に近いほど類似）")
                st.write(f"SSIM: {ssim_score:.4f}（1に近いほど類似）")
                st.write(f"検出BBOX数（上限{max_boxes}）: {len(bboxes)}")
                st.pyplot(fig, clear_figure=True)
        else:
            st.info("解析実行ボタンを押してください。")
