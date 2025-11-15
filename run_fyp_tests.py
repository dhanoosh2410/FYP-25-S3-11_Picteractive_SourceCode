import io
import json
import os
import uuid
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

# ---------------------------------------------------------------------------
# Environment / app wiring
# ---------------------------------------------------------------------------

# Seed a default admin user via auth_DB's import-time hook
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "AdminPass123!"
ADMIN_EMAIL = "admin@example.com"

os.environ.setdefault("ADMIN_USERNAME", ADMIN_USERNAME)
os.environ.setdefault("ADMIN_PASSWORD", ADMIN_PASSWORD)
os.environ.setdefault("ADMIN_EMAIL", ADMIN_EMAIL)

from server.main import app  # noqa: E402

ROOT = Path(__file__).resolve().parent
SRC_APP = ROOT / "src" / "App.jsx"
APP_SOURCE = SRC_APP.read_text(encoding="utf-8")


def make_client() -> TestClient:
    return TestClient(app)


def unique_user_creds(prefix: str = "user"):
    suf = uuid.uuid4().hex[:8]
    username = f"{prefix}_{suf}"
    email = f"{prefix}_{suf}@gmail.com"
    password = "TestPass123!"
    return username, email, password


def register_and_login(client: TestClient, prefix: str):
    username, email, password = unique_user_creds(prefix)
    r = client.post(
        "/api/auth/register",
        json={"username": username, "email": email, "password": password},
    )
    assert r.status_code == 200, f"register failed: {r.status_code} {r.text}"
    return {"username": username, "email": email, "password": password}


def sample_image_bytes(name: str = "scene1.jpg") -> bytes:
    img_path = ROOT / "src" / "assets" / name
    if not img_path.exists():
        raise AssertionError(f"sample image not found at {img_path}")
    return img_path.read_bytes()


# ---------------------------------------------------------------------------
# Tests (1:1 with the 50 scenarios; no IDs in names/text)
# ---------------------------------------------------------------------------


def test_01_account_creation_ok():
    client = make_client()
    creds = register_and_login(client, "acct_ok")
    assert client.cookies.get("session_id"), "session cookie missing after register"
    me = client.get("/api/auth/me")
    assert me.status_code == 200, f"/api/auth/me failed: {me.status_code}"
    data = me.json()
    assert data["username"] == creds["username"]
    assert data["email"] == creds["email"]


def test_02_account_creation_errors():
    client = make_client()

    # Missing fields
    r = client.post(
        "/api/auth/register", json={"username": "", "email": "", "password": ""}
    )
    assert r.status_code == 400
    assert "Missing required fields" in r.text

    # Duplicate username / email
    u1, e1, p1 = unique_user_creds("acct_err")
    r1 = client.post(
        "/api/auth/register", json={"username": u1, "email": e1, "password": p1}
    )
    assert r1.status_code == 200

    # Duplicate username with different email
    r2 = client.post(
        "/api/auth/register",
        json={"username": u1, "email": f"other_{e1}", "password": p1},
    )
    assert r2.status_code == 400
    assert "Username already exists" in r2.text

    # Duplicate email with different username
    u2, _e2, p2 = unique_user_creds("acct_err2")
    r3 = client.post(
        "/api/auth/register", json={"username": u2, "email": e1, "password": p2}
    )
    assert r3.status_code == 400
    assert "Email already exists" in r3.text


def test_03_admin_login_success():
    client = make_client()
    r = client.post(
        "/api/auth/login",
        json={"username_or_email": ADMIN_EMAIL, "password": ADMIN_PASSWORD},
    )
    assert r.status_code == 200, f"admin login failed: {r.status_code} {r.text}"
    assert client.cookies.get("session_id")


def test_04_admin_login_wrong_password():
    client = make_client()
    r = client.post(
        "/api/auth/login",
        json={"username_or_email": ADMIN_EMAIL, "password": ADMIN_PASSWORD + "x"},
    )
    assert r.status_code == 401


def test_05_user_login_success():
    client = make_client()
    u, e, p = unique_user_creds("login_ok")
    r = client.post(
        "/api/auth/register", json={"username": u, "email": e, "password": p}
    )
    assert r.status_code == 200

    client2 = make_client()
    r2 = client2.post(
        "/api/auth/login", json={"username_or_email": u, "password": p}
    )
    assert r2.status_code == 200, f"login failed: {r2.status_code} {r2.text}"
    assert client2.cookies.get("session_id")
    me = client2.get("/api/auth/me")
    assert me.status_code == 200


def test_06_user_login_wrong_credentials():
    client = make_client()
    u, e, p = unique_user_creds("login_bad")
    r = client.post(
        "/api/auth/register", json={"username": u, "email": e, "password": p}
    )
    assert r.status_code == 200

    client2 = make_client()
    r2 = client2.post(
        "/api/auth/login", json={"username_or_email": u, "password": "wrong-pass"}
    )
    assert r2.status_code == 401


def test_07_upload_image_accepts_image():
    client = make_client()
    raw = sample_image_bytes("scene1.jpg")
    files = {"image": ("scene1.jpg", raw, "image/jpeg")}
    data = {"mode": "detailed"}
    r = client.post("/api/caption", files=files, data=data)
    assert r.status_code == 200, f"/api/caption failed: {r.status_code} {r.text}"
    payload = r.json()
    assert "caption" in payload


def test_08_view_description_ui():
    assert "wt-readout" in APP_SOURCE
    assert "GENERATE DESCRIPTION" in APP_SOURCE


def test_09_region_selection():
    client = make_client()
    raw = sample_image_bytes("scene1.jpg")
    region = json.dumps({"x": 10, "y": 10, "w": 50, "h": 50})
    files = {"image": ("scene1.jpg", raw, "image/jpeg")}
    data = {"mode": "detailed", "region": region}
    r = client.post("/api/caption", files=files, data=data)
    assert r.status_code == 200, f"/api/caption with region failed: {r.status_code}"
    assert "REGION SELECT" in APP_SOURCE
    assert "RegionSelector" in APP_SOURCE


def test_10_camera_capture_ui():
    assert "navigator.mediaDevices.getUserMedia" in APP_SOURCE
    assert "TAKE A PHOTO" in APP_SOURCE
    assert "captureFrame(" in APP_SOURCE


def test_11_draw_page_exists():
    assert "function DrawPage()" in APP_SOURCE
    assert "new fabric.Canvas" in APP_SOURCE


def test_12_brush_tool_ui():
    assert "setTool('brush')" in APP_SOURCE
    assert ">Brush<" in APP_SOURCE or "Brush\n" in APP_SOURCE


def test_13_eraser_tool_ui():
    assert "setTool('eraser')" in APP_SOURCE
    assert ">Eraser<" in APP_SOURCE or "Eraser\n" in APP_SOURCE


def test_14_fill_tool_ui():
    assert "setTool('fill')" in APP_SOURCE
    assert ">Fill<" in APP_SOURCE or "Fill\n" in APP_SOURCE
    assert "floodFillAt(" in APP_SOURCE


def test_15_colour_picker_ui():
    assert 'type="color"' in APP_SOURCE or "type=\"color\"" in APP_SOURCE
    assert "draw-color-picker" in APP_SOURCE


def test_16_undo_redo_clear_ui():
    assert "handleUndo" in APP_SOURCE
    assert "handleRedo" in APP_SOURCE
    assert "handleClear" in APP_SOURCE
    assert ">Undo<" in APP_SOURCE or "Undo</button>" in APP_SOURCE
    assert ">Redo<" in APP_SOURCE or "Redo</button>" in APP_SOURCE
    assert ">Clear<" in APP_SOURCE or "Clear</button>" in APP_SOURCE


def test_17_frames_ui():
    assert "const [frames, setFrames]" in APP_SOURCE
    assert "ADD TO STORY" in APP_SOURCE
    assert "TRY EXAMPLE" in APP_SOURCE
    assert "CREATE STORY" in APP_SOURCE


def test_18_text_to_speech_api():
    client = make_client()
    r = client.post(
        "/api/tts", json={"text": "Hello world", "voice": "male", "rate": 1.0}
    )
    assert r.status_code == 200, f"/api/tts failed: {r.status_code} {r.text}"
    ctype = r.headers.get("content-type", "")
    assert ctype.startswith("audio/")


def test_19_quiz_generation_api():
    client = make_client()
    payload = {"caption": "A boy with a red ball", "count": 3}
    r = client.post("/api/quiz", json=payload)
    assert r.status_code == 200, f"/api/quiz failed: {r.status_code} {r.text}"
    data = r.json()
    qs = data.get("questions") or []
    assert isinstance(qs, list)
    assert len(qs) == 3
    first = qs[0]
    assert "question" in first and "options" in first


def test_20_dyslexia_font_setting_backend():
    client = make_client()
    creds = register_and_login(client, "dyslex")
    _ = creds
    me = client.get("/api/auth/me")
    assert me.status_code == 200
    data = me.json()
    settings = data.get("settings") or {}
    assert "dyslexia_font" in settings
    new_settings = dict(settings)
    new_settings["dyslexia_font"] = "Lexend"
    r = client.patch("/api/user/settings", json={"settings": new_settings})
    assert r.status_code == 200, f"/api/user/settings failed: {r.status_code} {r.text}"
    me2 = client.get("/api/auth/me")
    assert me2.status_code == 200
    data2 = me2.json()
    assert data2["settings"]["dyslexia_font"] == "Lexend"


def test_21_dictionary_api():
    client = make_client()
    r = client.get("/api/dictionary", params={"word": "apple"})
    assert r.status_code == 200, f"/api/dictionary failed: {r.status_code} {r.text}"
    data = r.json()
    assert "definition" in data
    assert "synonyms" in data
    assert "examples" in data


def test_22_colour_blind_condition_ui():
    assert "cbCond" in APP_SOURCE
    assert "COLOUR BLINDNESS SETTINGS" in APP_SOURCE
    assert "Protanopia (red-blind)" in APP_SOURCE
    assert "Deuteranopia (green-blind)" in APP_SOURCE


def test_23_colour_blind_severity_ui():
    assert "cbSeverity" in APP_SOURCE
    assert "Severity (mild" in APP_SOURCE
    assert 'type="range"' in APP_SOURCE or "type=\"range\"" in APP_SOURCE


def test_24_colour_blind_mode_ui():
    assert "cbMode" in APP_SOURCE
    assert "Simulate</label>" in APP_SOURCE
    assert "Enhance</label>" in APP_SOURCE
    assert "Split view" in APP_SOURCE


def test_25_translation_supported_languages():
    client = make_client()
    for lang in ("en", "zh", "ms", "ta"):
        r = client.post("/api/translate", json={"text": "Hello", "lang": lang})
        assert r.status_code == 200, f"/api/translate {lang} failed: {r.status_code}"
        data = r.json()
        assert data.get("lang") == lang
        assert "text" in data


def test_26_story_generation_api():
    client = make_client()
    img_dir = ROOT / "src" / "assets"
    data1 = (img_dir / "scene1.jpg").read_bytes()
    data2 = (img_dir / "scene2.jpg").read_bytes()
    data3 = (img_dir / "scene3.jpg").read_bytes()
    files = {
        "image1": ("scene1.jpg", data1, "image/jpeg"),
        "image2": ("scene2.jpg", data2, "image/jpeg"),
        "image3": ("scene3.jpg", data3, "image/jpeg"),
    }
    r = client.post("/api/story", files=files)
    assert r.status_code == 200, f"/api/story failed: {r.status_code} {r.text}"
    payload = r.json()
    panels = payload.get("panels") or []
    assert len(panels) == 3


def test_27_story_page_interactions_code():
    assert "STORY TIME!" in APP_SOURCE
    assert "story-title-input" in APP_SOURCE
    assert "DictionaryModal" in APP_SOURCE
    assert "onClick={downloadImage}" in APP_SOURCE or ".download = (safe || 'story')" in APP_SOURCE


def test_28_instructions_page_code():
    assert "function InstructionsPage()" in APP_SOURCE
    assert "INSTRUCTIONS" in APP_SOURCE
    assert "WELCOME TO PICTERACTIVE!" in APP_SOURCE


def test_29_achievements_api():
    client = make_client()
    register_and_login(client, "ach")
    r = client.post("/api/achievements/event", json={"type": "caption"})
    assert r.status_code == 200, f"/api/achievements/event failed: {r.status_code}"
    r2 = client.get("/api/me/progress")
    assert r2.status_code == 200
    prog = r2.json()
    assert "streak" in prog
    assert prog["streak"] >= 1
    r3 = client.get("/api/me/achievements")
    assert r3.status_code == 200
    badges = r3.json()
    assert isinstance(badges, list)


def test_30_profile_management_endpoints():
    client = make_client()
    creds = register_and_login(client, "profile")
    me = client.get("/api/auth/me")
    assert me.status_code == 200
    data = me.json()

    # Change display name
    new_name = data["username"] + "_x"
    r1 = client.post(
        "/api/account/change_display_name", json={"display_name": new_name}
    )
    assert r1.status_code == 200, f"change_display_name failed: {r1.status_code}"

    # Change email (Gmail only)
    new_email = "new_" + creds["username"] + "@gmail.com"
    r2 = client.post("/api/account/change_email", json={"email": new_email})
    assert r2.status_code == 200, f"change_email failed: {r2.status_code} {r2.text}"

    # Clear data
    r3 = client.post(
        "/api/account/clear_data", json={"password": creds["password"]}
    )
    assert r3.status_code == 200, f"clear_data failed: {r3.status_code}"

    me_after = client.get("/api/auth/me")
    assert me_after.status_code == 200
    d2 = me_after.json()
    ach = d2.get("achievements") or {}
    assert int(ach.get("streak_days", 0)) == 0

    # Delete account
    r4 = client.post(
        "/api/account/delete", json={"password": creds["password"]}
    )
    assert r4.status_code == 200, f"delete_account failed: {r4.status_code}"
    me_deleted = client.get("/api/auth/me")
    assert me_deleted.status_code == 401


def test_31_accessibility_options_persist():
    client = make_client()
    register_and_login(client, "access")
    me = client.get("/api/auth/me")
    assert me.status_code == 200
    settings = (me.json().get("settings") or {}).copy()
    # Flip some options
    settings["high_contrast"] = not bool(settings.get("high_contrast", False))
    settings["grid_guides"] = not bool(settings.get("grid_guides", False))
    settings["tts_voice"] = "male"
    settings["speaking_rate"] = 1.5
    settings["word_highlight_color"] = "#0000ff"
    r = client.patch("/api/user/settings", json={"settings": settings})
    assert r.status_code == 200, f"/api/user/settings failed: {r.status_code}"
    me2 = client.get("/api/auth/me")
    assert me2.status_code == 200
    s2 = me2.json().get("settings") or {}
    assert s2["high_contrast"] == settings["high_contrast"]
    assert s2["grid_guides"] == settings["grid_guides"]
    assert str(s2["tts_voice"]).lower() in {"male", "female", "app voice"}


def test_32_password_strength_validated_in_ui():
    # Registration form client-side validation
    assert "At least 6 characters" in APP_SOURCE
    # Login form client-side validation
    assert "Password must be at least 6 characters" in APP_SOURCE


def test_33_duplicate_email_registration_api():
    client = make_client()
    u1, e1, p1 = unique_user_creds("dup")
    r1 = client.post(
        "/api/auth/register", json={"username": u1, "email": e1, "password": p1}
    )
    assert r1.status_code == 200
    u2, _e2, p2 = unique_user_creds("dup2")
    r2 = client.post(
        "/api/auth/register", json={"username": u2, "email": e1, "password": p2}
    )
    assert r2.status_code == 400
    assert "Email already exists" in r2.text


def test_34_change_password_flow():
    client = make_client()
    creds = register_and_login(client, "pwchange")

    # Wrong current password
    r1 = client.post(
        "/api/account/change_password",
        json={"current_password": "wrong", "new_password": "NewPass123!"},
    )
    assert r1.status_code == 403

    # Correct change
    new_pw = "NewPass123!"
    r2 = client.post(
        "/api/account/change_password",
        json={"current_password": creds["password"], "new_password": new_pw},
    )
    assert r2.status_code == 200

    # Old password should fail; new password should work
    client2 = make_client()
    r_old = client2.post(
        "/api/auth/login",
        json={"username_or_email": creds["username"], "password": creds["password"]},
    )
    assert r_old.status_code == 401
    r_new = client2.post(
        "/api/auth/login",
        json={"username_or_email": creds["username"], "password": new_pw},
    )
    assert r_new.status_code == 200


def test_35_session_persistence_and_logout():
    client = make_client()
    creds = register_and_login(client, "session")
    _ = creds
    me1 = client.get("/api/auth/me")
    assert me1.status_code == 200

    # Session persists across multiple requests
    me2 = client.get("/api/auth/me")
    assert me2.status_code == 200

    # Logout clears session
    r = client.post("/api/auth/logout")
    assert r.status_code == 200
    me3 = client.get("/api/auth/me")
    assert me3.status_code == 401


def test_36_session_cookie_reuse_as_remember_me():
    client = make_client()
    creds = register_and_login(client, "remember")
    _ = creds
    assert client.cookies.get("session_id")
    sid = client.cookies.get("session_id")

    # Simulate a new browser instance reusing the cookie
    client2 = make_client()
    client2.cookies.set("session_id", sid, path="/")
    me = client2.get("/api/auth/me")
    assert me.status_code == 200


def test_37_invalid_image_rejected():
    client = make_client()
    files = {"image": ("bad.txt", b"not-an-image", "text/plain")}
    r = client.post("/api/caption", files=files)
    assert r.status_code == 400
    data = r.json()
    assert "error" in data
    assert "invalid_image" in data["error"]


def test_38_large_image_handled():
    client = make_client()
    # Create a moderately large in-memory image
    img = Image.new("RGB", (2048, 2048), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    files = {"image": ("large.jpg", buf.getvalue(), "image/jpeg")}
    r = client.post("/api/caption", files=files)
    # Either a caption or a graceful error (but not a crash)
    assert r.status_code in (200, 400)


def test_39_upload_cancel_code():
    # File input is wired only via onChange; cancelling the dialog leaves state unchanged.
    assert "type=\"file\"" in APP_SOURCE or 'type="file"' in APP_SOURCE
    assert "onChange={onFile}" in APP_SOURCE


def test_40_regenerate_caption_button_present():
    # Single generate button that can be clicked multiple times
    assert "GENERATE DESCRIPTION" in APP_SOURCE
    assert "onClick={()=>doCaption()}" in APP_SOURCE or "onClick={() => doCaption()}" in APP_SOURCE


def test_41_region_selection_cancel_code():
    assert "RegionSelector" in APP_SOURCE
    assert "onCancel={()=>setRegionOpen(false)}" in APP_SOURCE or "onCancel={() => setRegionOpen(false)}" in APP_SOURCE
    assert ">Cancel</button>" in APP_SOURCE


def test_42_camera_permission_error_handling_code():
    assert "NotAllowedError" in APP_SOURCE
    assert "Access blocked by browser/OS" in APP_SOURCE
    assert "Camera busy in another app" in APP_SOURCE


def test_43_save_item_creates_file():
    client = make_client()
    caption = "Test caption"
    raw = sample_image_bytes("scene1.jpg")
    files = {"image": ("scene1.jpg", raw, "image/jpeg")}
    data = {"caption": caption}
    r = client.post("/api/save", files=files, data=data)
    assert r.status_code == 200, f"/api/save failed: {r.status_code} {r.text}"
    item = r.json()
    assert "imageUrl" in item
    name = item["imageUrl"].split("/")[-1]
    img_path = ROOT / "server" / "data" / "scenes" / name
    assert img_path.exists()


def test_44_recent_and_serve_image_endpoints():
    client = make_client()
    # Ensure at least one item exists
    caption = "Recent test"
    raw = sample_image_bytes("scene1.jpg")
    files = {"image": ("scene1.jpg", raw, "image/jpeg")}
    data = {"caption": caption}
    r_save = client.post("/api/save", files=files, data=data)
    assert r_save.status_code == 200

    r_recent = client.get("/api/recent")
    assert r_recent.status_code == 200
    recent = r_recent.json()
    assert recent.get("caption") == caption
    image_url = recent.get("imageUrl") or ""
    name = image_url.split("/")[-1]
    r_img = client.get(f"/api/image/{name}")
    assert r_img.status_code == 200


def test_45_draw_frame_delete_code():
    assert "removeFrame(id)" in APP_SOURCE or "removeFrame(f.id)" in APP_SOURCE
    assert "draw-frame-remove" in APP_SOURCE


def test_46_quiz_feedback_messages_code():
    assert "quiz-feedback" in APP_SOURCE
    assert "Correct!" in APP_SOURCE
    assert "Not quite. Try the next one!" in APP_SOURCE


def test_47_dyslexia_font_dom_attributes_code():
    assert "applyAccessibility" in APP_SOURCE
    assert 'data-font' in APP_SOURCE
    assert 'data-contrast' in APP_SOURCE


def test_48_translation_unsupported_language_error():
    client = make_client()
    # lang is validated by the schema; unsupported values yield 422
    r = client.post("/api/translate", json={"text": "Hello", "lang": "fr"})
    assert r.status_code == 422


def test_49_story_save_and_list_endpoints():
    client = make_client()
    register_and_login(client, "storysave")
    payload = {
        "title": "My Test Story",
        "panels": ["Panel one", "Panel two", "Panel three"],
        "images": ["/api/image/a.jpg", "/api/image/b.jpg", "/api/image/c.jpg"],
        "story": "Panel one\nPanel two\nPanel three",
    }
    r = client.post("/api/stories/save", json=payload)
    assert r.status_code == 200, f"/api/stories/save failed: {r.status_code} {r.text}"
    saved = r.json()
    sid = saved.get("id")
    assert sid

    r_list = client.get("/api/stories/list")
    assert r_list.status_code == 200
    lst = r_list.json()
    assert any(item.get("id") == sid for item in lst)

    r_get = client.get("/api/stories/get", params={"id": sid})
    assert r_get.status_code == 200
    story = r_get.json().get("story") or {}
    assert story.get("title") == "My Test Story"


def test_50_accessibility_reset_to_defaults():
    client = make_client()
    register_and_login(client, "reset")
    me = client.get("/api/auth/me")
    assert me.status_code == 200
    base_settings = (me.json().get("settings") or {}).copy()

    # Change a couple of fields away from defaults
    changed = dict(base_settings)
    changed["high_contrast"] = not bool(base_settings.get("high_contrast", False))
    changed["dyslexia_font"] = (
        "OpenDyslexic"
        if base_settings.get("dyslexia_font") != "OpenDyslexic"
        else "Off"
    )
    r1 = client.patch("/api/user/settings", json={"settings": changed})
    assert r1.status_code == 200

    me_changed = client.get("/api/auth/me")
    assert me_changed.status_code == 200
    s1 = me_changed.json().get("settings") or {}
    assert s1["high_contrast"] == changed["high_contrast"]
    assert s1["dyslexia_font"] == changed["dyslexia_font"]

    # "Reset" by sending the original settings back
    r2 = client.patch("/api/user/settings", json={"settings": base_settings})
    assert r2.status_code == 200
    me_reset = client.get("/api/auth/me")
    assert me_reset.status_code == 200
    s2 = me_reset.json().get("settings") or {}
    assert s2["high_contrast"] == base_settings.get("high_contrast", False)
    assert s2["dyslexia_font"] == base_settings.get("dyslexia_font", "Off")


def main():
    tests = [
        ("User Management: account creation", test_01_account_creation_ok),
        ("User Management: account creation errors", test_02_account_creation_errors),
        ("Admin Login: correct credentials", test_03_admin_login_success),
        ("Admin Login: wrong credentials", test_04_admin_login_wrong_password),
        ("User Login: correct credentials", test_05_user_login_success),
        ("User Login: wrong credentials", test_06_user_login_wrong_credentials),
        ("Image: uploading image to caption API", test_07_upload_image_accepts_image),
        ("Image: description UI binding", test_08_view_description_ui),
        ("Image: region selection support", test_09_region_selection),
        ("Image: camera capture UI", test_10_camera_capture_ui),
        ("Drawing: page and canvas setup", test_11_draw_page_exists),
        ("Drawing: brush tool", test_12_brush_tool_ui),
        ("Drawing: eraser tool", test_13_eraser_tool_ui),
        ("Drawing: fill tool", test_14_fill_tool_ui),
        ("Drawing: colour picker", test_15_colour_picker_ui),
        ("Drawing: undo/redo/clear", test_16_undo_redo_clear_ui),
        ("Drawing: frames UI", test_17_frames_ui),
        ("Educational: text-to-speech API", test_18_text_to_speech_api),
        ("Educational: quiz generation API", test_19_quiz_generation_api),
        ("Educational: dyslexia font setting (backend)", test_20_dyslexia_font_setting_backend),
        ("Educational: dictionary API", test_21_dictionary_api),
        ("Educational: colour blindness condition", test_22_colour_blind_condition_ui),
        ("Educational: colour blindness severity", test_23_colour_blind_severity_ui),
        ("Educational: colour blindness mode", test_24_colour_blind_mode_ui),
        ("Educational: translation API supported languages", test_25_translation_supported_languages),
        ("Educational: story generation API", test_26_story_generation_api),
        ("Educational: story page interactions code", test_27_story_page_interactions_code),
        ("Instructions: instructions page code", test_28_instructions_page_code),
        ("Achievements: API and streak", test_29_achievements_api),
        ("Settings: profile management", test_30_profile_management_endpoints),
        ("Settings: accessibility options persistence", test_31_accessibility_options_persist),
        ("User Management: password strength UI validation", test_32_password_strength_validated_in_ui),
        ("User Management: duplicate email registration", test_33_duplicate_email_registration_api),
        ("User Management: change password flow", test_34_change_password_flow),
        ("User Login: session persistence and logout", test_35_session_persistence_and_logout),
        ("User Login: session cookie reuse", test_36_session_cookie_reuse_as_remember_me),
        ("Image: unsupported file rejected", test_37_invalid_image_rejected),
        ("Image: large image handled", test_38_large_image_handled),
        ("Image: upload cancel wiring", test_39_upload_cancel_code),
        ("Image: regenerate caption button", test_40_regenerate_caption_button_present),
        ("Image: region selection cancel wiring", test_41_region_selection_cancel_code),
        ("Image: camera permission error handling code", test_42_camera_permission_error_handling_code),
        ("Drawing: save item endpoint", test_43_save_item_creates_file),
        ("Drawing: load recent and serve image", test_44_recent_and_serve_image_endpoints),
        ("Drawing: delete frame UI", test_45_draw_frame_delete_code),
        ("Educational: quiz feedback messages", test_46_quiz_feedback_messages_code),
        ("Educational: dyslexia font DOM attributes", test_47_dyslexia_font_dom_attributes_code),
        ("Educational: translation unsupported language error", test_48_translation_unsupported_language_error),
        ("Educational: story save/list endpoints", test_49_story_save_and_list_endpoints),
        ("Settings: accessibility reset to defaults", test_50_accessibility_reset_to_defaults),
    ]

    passed = 0
    total = len(tests)
    for name, func in tests:
        try:
            func()
            print(f"[PASS] {name}")
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {name}: {e}")
        except Exception as e:
            print(f"[ERROR] {name}: {e}")

    print(f"\nSummary: {passed}/{total} tests passed.")


if __name__ == "__main__":
    main()

