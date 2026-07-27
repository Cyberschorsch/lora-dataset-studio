"""render_canvas: the composited init image and its keep-mask.

The mask has to be an 8-bit RGB PNG whose RED channel carries the values, white
outside the head (regenerate the pose and background) and the keep value inside,
with a blurred boundary standing in for the feather core FeatherMask cannot do.
"""
from PIL import Image

from app.services import zimage_edit_helper as z


def _render(tmp_path, framing='body', aspect='3:4'):
    src = tmp_path / 'ref.png'
    Image.new('RGB', (512, 512), 'red').save(src)
    plan = z.plan_shot(framing, aspect, (512, 512), denoise=0.75)
    canvas, mask = tmp_path / 'c.png', tmp_path / 'm.png'
    z.render_canvas(str(src), plan, str(canvas), str(mask))
    return plan, canvas, mask


def test_writes_both_files_at_the_canvas_size(tmp_path):
    plan, canvas, mask = _render(tmp_path)
    with Image.open(canvas) as c, Image.open(mask) as m:
        assert c.size == plan['canvas']
        assert m.size == plan['canvas']
        # RGB, not L and not RGBA: LoadImageMask reads RED, and its alpha path
        # would invert the values.
        assert m.mode == 'RGB'


def test_reference_lands_on_the_planned_box(tmp_path):
    plan, canvas, _mask = _render(tmp_path)
    px, py, side = plan['paste']
    with Image.open(canvas) as c:
        assert c.getpixel((px + side // 2, py + side // 2))[0] > 200   # the red crop
        assert c.getpixel((2, c.height - 2)) == (128, 128, 128)        # grey scaffold


def test_mask_is_white_outside_and_keeps_the_head(tmp_path):
    plan, _canvas, mask = _render(tmp_path)
    cx, cy, _rx, _ry = plan['ellipse']
    want = round(plan['keep'] * 255)
    with Image.open(mask) as m:
        corner = m.getpixel((2, 2))[0]
        centre = m.getpixel((cx, cy))[0]
        assert corner >= 250, 'outside the head must be fully regenerated'
        assert abs(centre - want) <= 12, (centre, want)
        assert centre < corner


def test_the_boundary_is_feathered(tmp_path):
    """A hard edge between held and generated latent is exactly what seams."""
    plan, _canvas, mask = _render(tmp_path)
    cx, cy, rx, _ry = plan['ellipse']
    with Image.open(mask) as m:
        centre = m.getpixel((cx, cy))[0]
        edge = m.getpixel((min(m.width - 1, cx + rx), cy))[0]
        corner = m.getpixel((2, 2))[0]
    assert centre < edge < corner, (centre, edge, corner)


def test_full_frame_source_is_scaled_to_the_figure(tmp_path):
    """body_source='original' composites the whole frame, not the head square."""
    src = tmp_path / 'full.png'
    Image.new('RGB', (600, 900), 'blue').save(src)
    plan = z.plan_shot('body', '3:4', (600, 900), denoise=0.75,
                       body_source='original', has_full_frame=True)
    assert plan['source'] == 'original'
    canvas, mask = tmp_path / 'c.png', tmp_path / 'm.png'
    z.render_canvas(str(src), plan, str(canvas), str(mask))
    with Image.open(canvas) as c:
        assert c.size == plan['canvas']
        px, py, side = plan['paste']
        assert c.getpixel((px + side // 2, py + side // 2))[2] > 200
