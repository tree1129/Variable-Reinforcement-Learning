from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

SHAPE_ORDER = ("triangle", "square", "trapezoid", "sphere")
FACE_ORDER = ("front", "right", "left")
COLOR = {"triangle": "blue", "square": "red", "trapezoid": "green", "sphere": "yellow"}
SHAPE_EN = {"triangle": "triangular", "square": "square", "trapezoid": "trapezoidal", "sphere": "spherical"}
SHAPE_ZH = {"triangle": "三角形", "square": "方形", "trapezoid": "梯形", "sphere": "球形"}


@dataclass(frozen=True)
class TaskSpec:
    case_id: int
    box_position: str
    placements: Mapping[str, str]
    hole_faces: Mapping[str, str]
    targets: tuple[str, ...]
    excluded: str

    def validate(self) -> None:
        if set(self.placements) != set(SHAPE_ORDER):
            raise ValueError(f"Case {self.case_id}: placements must contain all four shapes")
        if set(self.hole_faces) != set(self.targets):
            raise ValueError(f"Case {self.case_id}: hole_faces and targets differ")
        if set(self.hole_faces.values()) != set(FACE_ORDER):
            raise ValueError(f"Case {self.case_id}: hole faces must be front/right/left")
        if self.excluded in self.targets:
            raise ValueError(f"Case {self.case_id}: excluded object is a target")


CASES = {
    # Placements deliberately pair every required object with the arm that can
    # physically serve its assigned sorter face: left-side sources -> left arm,
    # right-side sources -> right arm, while either arm may use the front face.
    4: TaskSpec(
        4, "center",
        {"triangle": "left_near", "square": "right_far", "trapezoid": "right_near", "sphere": "left_far"},
        {"triangle": "front", "square": "right", "sphere": "left"},
        ("triangle", "square", "sphere"), "trapezoid",
    ),
    5: TaskSpec(
        5, "middle_left",
        {"triangle": "left_middle", "square": "right_far", "trapezoid": "right_middle", "sphere": "left_near"},
        {"square": "front", "trapezoid": "right", "triangle": "left"},
        ("square", "trapezoid", "triangle"), "sphere",
    ),
    6: TaskSpec(
        6, "middle_right",
        {"triangle": "left_near", "square": "left_far", "trapezoid": "right_near", "sphere": "right_middle"},
        {"trapezoid": "front", "sphere": "right", "square": "left"},
        ("trapezoid", "sphere", "square"), "triangle",
    ),
    7: TaskSpec(
        7, "center",
        {"triangle": "right_far", "square": "left_far", "trapezoid": "left_middle", "sphere": "right_middle"},
        {"sphere": "front", "triangle": "right", "trapezoid": "left"},
        ("sphere", "triangle", "trapezoid"), "square",
    ),
}
for _task in CASES.values():
    _task.validate()


def get_task(case_id: int) -> TaskSpec:
    try:
        return CASES[int(case_id)]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"case_id must be 4, 5, 6, or 7; got {case_id!r}") from exc


def build_prompt(case_id: int, language: str = "en") -> str:
    """Return the stable short task instruction used by a policy."""
    task = get_task(case_id)
    if language == "zh":
        actions = [f"将{SHAPE_ZH[s]}积木投入投币盒{face}面的对应形状孔" for s, face in task.hole_faces.items()]
        return "，".join(actions) + f"。不要投入{SHAPE_ZH[task.excluded]}积木；盒体保持不动，完成后收回机械臂。"
    if language != "en":
        raise ValueError("language must be 'en' or 'zh'")
    actions = [f"the {COLOR[s]} {SHAPE_EN[s]} block into the matching hole on the {face} face" for s, face in task.hole_faces.items()]
    action_text = ", ".join(actions[:-1]) + ", and " + actions[-1]
    return (
        f"Insert {action_text}. Do not insert the {COLOR[task.excluded]} {SHAPE_EN[task.excluded]} block. "
        "Keep the coin box fixed, then retract the robot arm to the safe home pose."
    )
