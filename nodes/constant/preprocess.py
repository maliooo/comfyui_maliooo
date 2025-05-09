"""
if "CannyEdgePreprocessor" in nodes:
        control_net_preprocessors["canny"] = (
            nodes["CannyEdgePreprocessor"],
            [100, 200],
        )
    if "LineArtPreprocessor" in nodes:
        control_net_preprocessors["lineart"] = (
            nodes["LineArtPreprocessor"],
            ["disable"],
        )
        control_net_preprocessors["lineart_coarse"] = (
            nodes["LineArtPreprocessor"],
            ["enable"],
        )
    if "AnimeLineArtPreprocessor" in nodes:
        control_net_preprocessors["lineart_anime"] = (
            nodes["AnimeLineArtPreprocessor"],
            [],
        )
    if "Manga2Anime_LineArt_Preprocessor" in nodes:
        control_net_preprocessors["lineart_manga"] = (
            nodes["Manga2Anime_LineArt_Preprocessor"],
            [],
        )
    if "ScribblePreprocessor" in nodes:
        control_net_preprocessors["scribble"] = (nodes["ScribblePreprocessor"], [])
    if "FakeScribblePreprocessor" in nodes:
        control_net_preprocessors["scribble_hed"] = (
            nodes["FakeScribblePreprocessor"],
            ["enable"],
        )
    if "HEDPreprocessor" in nodes:
        control_net_preprocessors["hed"] = (nodes["HEDPreprocessor"], ["disable"])
        control_net_preprocessors["hed_safe"] = (nodes["HEDPreprocessor"], ["enable"])
    if "PiDiNetPreprocessor" in nodes:
        control_net_preprocessors["pidi"] = (
            nodes["PiDiNetPreprocessor"],
            ["disable"],
        )
        control_net_preprocessors["pidi_safe"] = (
            nodes["PiDiNetPreprocessor"],
            ["enable"],
        )
    if "M-LSDPreprocessor" in nodes:
        control_net_preprocessors["mlsd"] = (nodes["M-LSDPreprocessor"], [0.1, 0.1])
    if "OpenposePreprocessor" in nodes:
        control_net_preprocessors["openpose"] = (
            nodes["OpenposePreprocessor"],
            ["enable", "enable", "enable"],
        )
        control_net_preprocessors["pose"] = control_net_preprocessors["openpose"]
    if "DWPreprocessor" in nodes:
        control_net_preprocessors["dwpose"] = (
            nodes["DWPreprocessor"],
            ["enable", "enable", "enable", "yolox_l.onnx", "dw-ll_ucoco_384.onnx"],
        )
        # use DWPreprocessor for pose by default if available
        control_net_preprocessors["pose"] = control_net_preprocessors["dwpose"]
    if "BAE-NormalMapPreprocessor" in nodes:
        control_net_preprocessors["normalmap_bae"] = (
            nodes["BAE-NormalMapPreprocessor"],
            [],
        )
    if "MiDaS-NormalMapPreprocessor" in nodes:
        control_net_preprocessors["normalmap_midas"] = (
            nodes["MiDaS-NormalMapPreprocessor"],
            [math.pi * 2.0, 0.1],
        )
    if "MiDaS-DepthMapPreprocessor" in nodes:
        control_net_preprocessors["depth_midas"] = (
            nodes["MiDaS-DepthMapPreprocessor"],
            [math.pi * 2.0, 0.4],
        )
    if "Zoe-DepthMapPreprocessor" in nodes:
        control_net_preprocessors["depth"] = (nodes["Zoe-DepthMapPreprocessor"], [])
        control_net_preprocessors["depth_zoe"] = (nodes["Zoe-DepthMapPreprocessor"], [])
    if "OneFormer-COCO-SemSegPreprocessor" in nodes:
        control_net_preprocessors["seg_ofcoco"] = (
            nodes["OneFormer-COCO-SemSegPreprocessor"],
            [],
        )
    if "OneFormer-ADE20K-SemSegPreprocessor" in nodes:
        control_net_preprocessors["seg_ofade20k"] = (
            nodes["OneFormer-ADE20K-SemSegPreprocessor"],
            [],
        )
    if "UniFormer-SemSegPreprocessor" in nodes:
        control_net_preprocessors["seg_ufade20k"] = (
            nodes["UniFormer-SemSegPreprocessor"],
            [],
        )
"""


# webui 对应的预处理器
    # "module_list": [
    #     "none",
    #     "ip-adapter-auto",
    #     "tile_resample",
    #     "pidinet",
    #     "oneformer_ade20k",
    #     "pidinet_scribble",
    #     "revision_clipvision",
    #     "reference_only",
    #     "recolor_luminance",
    #     "openpose_full",
    #     "normal_bae",
    #     "mlsd",
    #     "lineart_standard",
    #     "ip-adapter_clip_sd15",
    #     "inpaint_only",
    #     "depth",
    #     "canny",
    #     "invert",
    #     "tile_colorfix+sharp",
    #     "tile_colorfix",
    #     "threshold",
    #     "clip_vision",
    #     "pidinet_sketch",
    #     "color",
    #     "te_hed",
    #     "pidinet_safe",
    #     "hed_safe",
    #     "hed",
    #     "shuffle",
    #     "segmentation",
    #     "oneformer_coco",
    #     "anime_face_segment",
    #     "scribble_xdog",
    #     "scribble_hed",
    #     "revision_ignore_prompt",
    #     "reference_adain+attn",
    #     "reference_adain",
    #     "recolor_intensity",
    #     "openpose_hand",
    #     "openpose_faceonly",
    #     "openpose_face",
    #     "openpose",
    #     "normal_map",
    #     "normal_dsine",
    #     "mediapipe_face",
    #     "lineart",
    #     "lineart_coarse",
    #     "lineart_anime_denoise",
    #     "lineart_anime",
    #     "ip-adapter_face_id_plus",
    #     "ip-adapter_face_id",
    #     "ip-adapter_clip_sdxl_plus_vith",
    #     "ip-adapter_clip_sdxl",
    #     "instant_id_face_keypoints",
    #     "instant_id_face_embedding",
    #     "inpaint_only+lama",
    #     "inpaint",
    #     "dw_openpose_full",
    #     "depth_zoe",
    #     "depth_leres++",
    #     "depth_leres",
    #     "depth_hand_refiner",
    #     "depth_anything",
    #     "densepose_parula",
    #     "densepose",
    #     "blur_gaussian",
    #     "animal_openpose"
    # ],

# custom_nodes\comfyui-art-venture\modules\controlnet\preprocessors.py
# av_controlnet 对应的预处理器 ["lineart", "lineart_coarse", "lineart_anime", "lineart_manga", "scribble", "scribble_hed", "hed", "hed_safe", "pidi", "pidi_safe", "mlsd", "openpose", "pose", "dwpose", "normalmap_bae", "normalmap_midas", "depth_midas", "depth", "depth_zoe", "seg_ofcoco", "seg_ofade20k", "seg_ufade20k"]

# {'depth_leres++', 'scribble_xdog', 'lineart_realistic', 'lineart_standard', 'ip-adapter_clip_sd15'}
WEBUI_2_COMFYUI_PREPROCESS = {
    "scribble_xdog": "scribble",
    "lineart_realistic": "lineart",
    "lineart_coarse" : "lineart_coarse",
    "lineart_standard" : "lineart_standard",
    "depth_leres++" : "depth_leres"
}