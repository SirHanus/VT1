"""
Test PDF generation from benchmark plots.
"""

import sys
from pathlib import Path

try:
    from PIL import Image
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.pdfgen import canvas
except ImportError:
    print("ERROR: Required packages not installed")
    print("Please install: pip install pillow reportlab")
    sys.exit(1)


def create_pdf(image_paths, output_path):
    """Create a PDF with one image per page."""

    # Determine page size based on first image aspect ratio
    first_img = Image.open(image_paths[0])
    img_width, img_height = first_img.size

    # Use A4 landscape for wide plots
    if img_width > img_height:
        pagesize = (842, 595)  # A4 landscape in points (72 points per inch)
    else:
        pagesize = A4

    print(f"Creating PDF with {len(image_paths)} plots...")
    print(f"Page size: {pagesize[0]:.0f}x{pagesize[1]:.0f} points")

    c = canvas.Canvas(str(output_path), pagesize=pagesize)
    page_width, page_height = pagesize

    for i, img_path in enumerate(image_paths):
        print(f"  Adding plot {i+1}/{len(image_paths)}: {Path(img_path).name}")

        # Load image
        img = Image.open(img_path)
        img_width, img_height = img.size

        # Calculate scaling to fit page with margins
        margin = 36  # 0.5 inch margin (72 points per inch)
        available_width = page_width - 2 * margin
        available_height = page_height - 2 * margin

        # Scale to fit
        scale_w = available_width / img_width
        scale_h = available_height / img_height
        scale = min(scale_w, scale_h)

        new_width = img_width * scale
        new_height = img_height * scale

        # Center on page
        x = (page_width - new_width) / 2
        y = (page_height - new_height) / 2

        # Draw image
        c.drawImage(
            str(img_path),
            x,
            y,
            width=new_width,
            height=new_height,
            preserveAspectRatio=True,
        )

        # Add page number at bottom right
        c.setFont("Helvetica", 10)
        c.drawString(page_width - 50, 20, f"{i+1}/{len(image_paths)}")

        # Add filename at bottom left
        c.setFont("Helvetica", 8)
        c.drawString(20, 20, Path(img_path).name)

        c.showPage()

    c.save()
    print(f"\n✅ PDF saved: {output_path}")
    print(f"   Pages: {len(image_paths)}")
    print(f"   Size: {Path(output_path).stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python create_benchmark_pdf.py <image1> <image2> ... <output.pdf>"
        )
        sys.exit(1)

    image_paths = sys.argv[1:-1]
    output_path = sys.argv[-1]

    if not image_paths:
        print("ERROR: No image files provided")
        sys.exit(1)

    # Verify all images exist
    for img_path in image_paths:
        if not Path(img_path).exists():
            print(f"ERROR: Image not found: {img_path}")
            sys.exit(1)

    create_pdf(image_paths, output_path)
