"""
独自例外クラス定義

画像対称性解析における各種エラーを表現する例外クラスを提供します。
"""


class SymmetryAnalysisError(Exception):
    """基底例外クラス"""

    def __init__(self, message, diagnostics=None):
        """
        Args:
            message: エラーメッセージ
            diagnostics: 診断情報（辞書形式）
        """
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class ImageLoadError(SymmetryAnalysisError):
    """画像読み込みエラー"""
    pass


class AlignmentError(SymmetryAnalysisError):
    """位置合わせエラー"""
    pass


class CroppingError(SymmetryAnalysisError):
    """画角切り出しエラー"""
    pass


class FeatureDetectionError(SymmetryAnalysisError):
    """特徴点検出エラー"""
    pass


class QualityCheckError(SymmetryAnalysisError):
    """画質チェックエラー"""
    pass


class BBoxDetectionError(SymmetryAnalysisError):
    """BBOX検出エラー"""
    pass


class PreprocessingError(SymmetryAnalysisError):
    """前処理エラー"""
    pass


class ConfigurationError(SymmetryAnalysisError):
    """設定エラー"""
    pass
