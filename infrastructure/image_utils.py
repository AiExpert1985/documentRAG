# infrastructure/image_utils.py
"""
Image processing utilities for thumbnails and optimization.
"""
import asyncio
import logging
from pathlib import Path
from typing import Tuple

from PIL import Image

from config import settings

logger = logging.getLogger(getattr(settings, "LOGGER_NAME", __name__))


class ImageProcessor:
    """Handle image optimization and thumbnail generation."""

    # Thumbnail constraints
    THUMB_MAX_DIMENSION = 1024  # Max width or height
    THUMB_QUALITY = 85

    # Original image settings
    ORIGINAL_COMPRESS_LEVEL = 6

    @staticmethod
    async def save_page_image_with_thumbnail(
        image: Image.Image,
        base_path: Path,
        filename_base: str
    ) -> Tuple[str, str]:
        """
        Save both original and thumbnail versions of page image.

        Args:
            image: PIL Image object
            base_path: Directory to save images
            filename_base: Base filename (e.g., "page_001")

        Returns:
            Tuple of (original_abs_path, thumbnail_abs_path)
        """
        base_path.mkdir(parents=True, exist_ok=True)

        original_path = base_path / f"{filename_base}.png"
        thumb_path = base_path / f"{filename_base}_thumb.webp"

        # Save original (PNG for fidelity)
        await asyncio.to_thread(
            image.save,
            str(original_path),
            format="PNG",
            optimize=True,
            compress_level=ImageProcessor.ORIGINAL_COMPRESS_LEVEL,
        )

        # Create and save thumbnail (WebP for size)
        thumbnail = await ImageProcessor._create_thumbnail(image)
        await asyncio.to_thread(
            thumbnail.save,
            str(thumb_path),
            format="WEBP",
            quality=ImageProcessor.THUMB_QUALITY,
            method=6,  # Better compression
        )

        try:
            orig_kb = original_path.stat().st_size / 1024
            thumb_kb = thumb_path.stat().st_size / 1024
            logger.info(
                f"[IMAGE] {filename_base}: original={orig_kb:.0f}KB, "
                f"thumbnail={thumb_kb:.0f}KB (saved {orig_kb - thumb_kb:.0f}KB)"
            )
        except Exception as e:
            logger.debug(f"[IMAGE] stat() failed for {filename_base}: {e}")

        return str(original_path), str(thumb_path)

    @staticmethod
    async def _create_thumbnail(image: Image.Image) -> Image.Image:
        """
        Create thumbnail respecting aspect ratio.
        Scales image so longest dimension = THUMB_MAX_DIMENSION
        """
        def resize():
            width, height = image.size
            if width == 0 or height == 0:
                return image.copy()

            if width > height:
                new_w = ImageProcessor.THUMB_MAX_DIMENSION
                new_h = int(height * (new_w / width))
            else:
                new_h = ImageProcessor.THUMB_MAX_DIMENSION
                new_w = int(width * (new_h / height))

            thumb = image.copy()
            thumb.thumbnail((new_w, new_h), Image.Resampling.LANCZOS)
            return thumb

        return await asyncio.to_thread(resize)
