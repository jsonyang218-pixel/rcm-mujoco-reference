#!/usr/bin/env python3
"""Export the runtime v9 eye scene as a portable MJCF file."""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from rcm_scene_v9 import RCMSceneV9

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "v9_eye_scene.xml")
    args = parser.parse_args(argv)

    scene = RCMSceneV9()
    try:
        root = ET.fromstring(scene.xml)
        compiler = root.find("compiler")
        if compiler is None:
            raise ValueError("runtime MJCF is missing compiler element")
        compiler.set("meshdir", "mujoco_model/assets")
        ET.indent(root, space="  ")
        xml_text = ET.tostring(root, encoding="unicode")

        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('<?xml version="1.0" encoding="utf-8"?>\n'
                          + xml_text + "\n", encoding="utf-8")

        exported = mujoco.MjModel.from_xml_path(str(output))
        for name in ("eye_reference", "retina_visual", "lens_visual",
                     "lesion_marker", "needle_tip_marker"):
            if mujoco.mj_name2id(
                    exported, mujoco.mjtObj.mjOBJ_GEOM, name) < 0:
                raise ValueError(f"exported scene is missing geom {name}")
        if mujoco.mj_name2id(
                exported, mujoco.mjtObj.mjOBJ_CAMERA, "eye_internal") < 0:
            raise ValueError("exported scene is missing eye_internal camera")
        print(output)
    finally:
        scene.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
