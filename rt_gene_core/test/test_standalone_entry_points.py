from pathlib import Path


def test_core_package_declares_ros_free_demo_entry_points():
    setup_py = Path(__file__).resolve().parents[1] / "setup.py"
    text = setup_py.read_text()

    assert "rt-gene-demo-image = rt_gene.single_image_demo:main" in text
    assert "rt-bene-demo-blink = rt_bene.single_image_demo:main" in text
