import os
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import layout_manager
from layout_manager import (
    Region, ResolutionOverride, compose, compute_pip_region, get_canvas_size,
    parse_instance_ref, render_plugin_instance,
)


class FakeDeviceConfig:
    def __init__(self, resolution=(800, 480), orientation="horizontal", instances=None, plugins=None):
        self.resolution = resolution
        self.orientation = orientation
        self.instances = instances or {}
        self.plugins = plugins or {}
        self.plugin_image_dir = "/tmp"

    def get_resolution(self):
        return self.resolution

    def get_config(self, key=None, default=None):
        return {"orientation": self.orientation}.get(key, default)

    def get_playlist_manager(self):
        return self

    def find_plugin(self, plugin_id, name):
        return self.instances.get((plugin_id, name))

    def get_plugin(self, plugin_id):
        return self.plugins.get(plugin_id)


class FakeInstance:
    def __init__(self, settings):
        self.settings = settings


class SolidPlugin:
    """Renders a solid colour at whatever resolution the device config reports."""
    def __init__(self, config):
        self.config = config
        self.calls = []

    def generate_image(self, settings, device_config):
        dims = device_config.get_resolution()
        if device_config.get_config("orientation") == "vertical":
            dims = dims[::-1]
        self.calls.append(dims)
        return Image.new("RGB", dims, settings["color"])


class TestComputePipRegion:

    @pytest.mark.parametrize(
        "corner,expected",
        [
            ("top-left", Region(10, 10, 200, 120)),
            ("top-right", Region(590, 10, 200, 120)),
            ("bottom-left", Region(10, 350, 200, 120)),
            ("bottom-right", Region(590, 350, 200, 120)),
        ],
    )
    def test_corners(self, corner, expected):
        assert compute_pip_region((800, 480), corner, 25, margin=10) == expected

    def test_full_size_ignores_margin(self):
        assert compute_pip_region((800, 480), "bottom-right", 100, margin=20) == Region(0, 0, 800, 480)

    def test_size_is_clamped(self):
        assert compute_pip_region((800, 480), "top-left", 250).size == (800, 480)
        assert compute_pip_region((800, 480), "top-left", 0).size == (8, 5)

    def test_invalid_corner(self):
        with pytest.raises(ValueError):
            compute_pip_region((800, 480), "middle", 25)


class TestResolutionOverride:

    def test_horizontal(self):
        config = ResolutionOverride(FakeDeviceConfig(), (200, 120))
        assert config.get_resolution() == (200, 120)
        assert config.plugin_image_dir == "/tmp"

    def test_vertical_is_stored_pre_rotation(self):
        device_config = FakeDeviceConfig(orientation="vertical")
        assert get_canvas_size(device_config) == (480, 800)
        config = ResolutionOverride(device_config, (120, 200))
        # plugins flip the resolution for vertical displays, landing on the region size
        assert config.get_resolution()[::-1] == (120, 200)


class TestRenderAndCompose:

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        self.plugin = SolidPlugin({"id": "solid"})
        monkeypatch.setattr(layout_manager, "get_plugin_instance", lambda config: self.plugin)
        self.device_config = FakeDeviceConfig(
            instances={("solid", "Red"): FakeInstance({"color": "red"}),
                       ("solid", "Blue"): FakeInstance({"color": "blue"})},
            plugins={"solid": {"id": "solid"}},
        )

    def test_parse_instance_ref(self):
        assert parse_instance_ref("weather/Home Weather") == ("weather", "Home Weather")
        with pytest.raises(RuntimeError):
            parse_instance_ref("")

    def test_renders_at_region_size(self):
        image = render_plugin_instance(self.device_config, "solid/Red", (200, 120))
        assert image.size == (200, 120)
        assert self.plugin.calls == [(200, 120)]

    def test_renders_large_and_scales_down(self):
        image = render_plugin_instance(self.device_config, "solid/Red", (200, 120), render_size=(800, 480))
        assert image.size == (200, 120)
        assert self.plugin.calls == [(800, 480)]

    def test_missing_instance(self):
        with pytest.raises(RuntimeError, match="no longer exists"):
            render_plugin_instance(self.device_config, "solid/Green", (200, 120))

    def test_compose_places_overlay_with_border(self):
        region = compute_pip_region((800, 480), "bottom-right", 25, margin=10)
        background = render_plugin_instance(self.device_config, "solid/Red", (800, 480))
        overlay = render_plugin_instance(self.device_config, "solid/Blue", region.size)

        canvas = compose((800, 480), [(background, Region(0, 0, 800, 480)), (overlay, region)], border_width=2)

        assert canvas.size == (800, 480)
        assert canvas.getpixel((10, 10)) == (255, 0, 0)
        assert canvas.getpixel((region.x + 50, region.y + 50)) == (0, 0, 255)
        assert canvas.getpixel((region.x, region.y)) == (0, 0, 0)
        assert canvas.getpixel((789, 469)) == (0, 0, 0)
        assert canvas.getpixel((795, 475)) == (255, 0, 0)


class TestTextOverlay:

    @pytest.fixture
    def font(self):
        from utils.app_utils import get_font
        return get_font("Jost", 40, "bold")

    @pytest.mark.parametrize(
        "position,align",
        [(p, a) for p in layout_manager.TEXT_POSITIONS for a in layout_manager.TEXT_ALIGNMENTS],
    )
    def test_box_placement(self, font, position, align):
        canvas = Image.new("RGB", (800, 480), "white")
        box = layout_manager.draw_text_overlay(
            canvas, "Walk: Sam", font, position=position, align=align,
            background="#ff0000", padding=6, margin=10)

        assert box.y == (10 if position == "top" else 480 - box.height - 10)
        expected_x = {"left": 10, "right": 800 - box.width - 10, "center": (800 - box.width) // 2}[align]
        assert box.x == expected_x
        # the box is filled and the text is drawn inside it
        assert canvas.getpixel((box.x + 1, box.y + 1)) == (255, 0, 0)
        inside = canvas.crop((box.x, box.y, box.x + box.width, box.y + box.height))
        assert (0, 0, 0) in {colour for _, colour in inside.getcolors(maxcolors=100000)}

    def test_multiline_text_is_taller(self, font):
        canvas = Image.new("RGB", (800, 480), "white")
        one = layout_manager.draw_text_overlay(canvas, "Today: Sam", font)
        two = layout_manager.draw_text_overlay(canvas, "Today: Sam\nTomorrow: Alex", font)
        assert two.height > one.height

    def test_no_background_leaves_canvas_untouched_around_text(self, font):
        canvas = Image.new("RGB", (800, 480), "blue")
        box = layout_manager.draw_text_overlay(canvas, "Sam", font, background=None, padding=6)
        assert canvas.getpixel((box.x + 1, box.y + 1)) == (0, 0, 255)

    def test_invalid_position(self, font):
        with pytest.raises(ValueError):
            layout_manager.draw_text_overlay(Image.new("RGB", (10, 10)), "x", font, position="middle")


class TestFetchOverlayText:

    class FakeResponse:
        """Mimics requests: .text decodes .content with .encoding, defaulting to ISO-8859-1."""
        def __init__(self, text, status=200, content_type="text/plain"):
            self.content = text.encode("utf-8")
            self.status = status
            self.headers = {"Content-Type": content_type}
            self.encoding = "ISO-8859-1"

        @property
        def text(self):
            return self.content.decode(self.encoding)

        def raise_for_status(self):
            if self.status >= 400:
                raise layout_manager.requests.HTTPError(f"{self.status}")

    def _session(self, monkeypatch, response=None, error=None):
        class FakeSession:
            def get(self, url, timeout):
                if error:
                    raise error
                return response
        monkeypatch.setattr(layout_manager, "get_http_session", lambda: FakeSession())

    def test_returns_stripped_text(self, monkeypatch):
        self._session(monkeypatch, self.FakeResponse("  Walk: Sam\n"))
        assert layout_manager.fetch_overlay_text("http://x") == "Walk: Sam"

    def test_utf8_without_charset(self, monkeypatch):
        self._session(monkeypatch, self.FakeResponse("Sam · Zoë 🐕"))
        assert layout_manager.fetch_overlay_text("http://x") == "Sam · Zoë 🐕"

    def test_declared_charset_is_respected(self, monkeypatch):
        response = self.FakeResponse("Sam", content_type="text/plain; charset=ISO-8859-1")
        self._session(monkeypatch, response)
        layout_manager.fetch_overlay_text("http://x")
        assert response.encoding == "ISO-8859-1"

    def test_truncates_long_text(self, monkeypatch):
        self._session(monkeypatch, self.FakeResponse("a" * 2000))
        assert len(layout_manager.fetch_overlay_text("http://x")) == layout_manager.MAX_OVERLAY_TEXT_LENGTH

    def test_http_error_returns_none(self, monkeypatch):
        self._session(monkeypatch, self.FakeResponse("Not found", status=404))
        assert layout_manager.fetch_overlay_text("http://x") is None

    def test_connection_error_returns_none(self, monkeypatch):
        self._session(monkeypatch, error=layout_manager.requests.ConnectionError("down"))
        assert layout_manager.fetch_overlay_text("http://x") is None
