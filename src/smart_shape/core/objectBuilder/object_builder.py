from PyQt5.QtGui import (
    QPaintEvent,
    QPainter,
    QPen,
    QBrush,
    QColor,
    QFont,
    QPolygonF,
    QCursor,
    QPainterPath,
    QTransform,
)
from PyQt5.QtWidgets import QWidget
from taurus.qt.qtgui.base import TaurusBaseComponent
from PyQt5.QtCore import Qt, QPointF, pyqtSignal, pyqtSlot, QRect
from PyQt5.QtCore import QTimer, QObject
import numpy as np
import copy
import math
from functools import partial
import time
import yaml
from dataclasses import dataclass
from ..callbacks.callback_container import *
from ...util.util import findMainWindow
from ..shapes.composite_shape import shapeComposite
from ..shapes.basic_shape import *
from smart_shape.util.geometry_transformation import (
    rotate_multiple_points,
    angle_between,
    line_intersection,
)

DECORATION_UPON_CURSOR_ON = {
    "pen": {"color": (255, 255, 0), "width": 1, "ls": "DotLine"},
    "brush": {"color": (0, 0, 255, 255)},
}
DECORATION_UPON_CURSOR_OFF = {
    "pen": {"color": (255, 0, 0), "width": 1, "ls": "SolidLine"},
    "brush": {"color": (0, 0, 255, 255)},
}

DECORATION_TEXT_DEFAULT = {
    "font_size": 10,
    "text_color": (255, 255, 255),
    "alignment": "AlignCenter",
    "padding": 0,
}


def make_decoration_from_text(
    dec={
        "pen": {"color": (255, 255, 0), "width": 3, "ls": "DotLine"},
        "brush": {"color": (0, 0, 255)},
    }
):

    pen_color = QColor(*dec["pen"]["color"])
    pen_width = dec["pen"]["width"]
    pen_style = getattr(Qt, dec["pen"]["ls"])
    qpen = QPen(pen_color, pen_width, pen_style)
    brush_color = QColor(*dec["brush"]["color"])
    qbrush = QBrush(brush_color)
    return {"pen": qpen, "brush": qbrush}


class buildTools(object):

    @classmethod
    def build_basic_shape_from_yaml(cls, yaml_file_path):
        with open(yaml_file_path, "r", encoding="utf8") as f:
            config = yaml.safe_load(f.read())
        shape_container = {}
        basic_shapes = config["basic_shapes"]
        for shape, shape_info in basic_shapes.items():
            for shape_type, shape_type_info in shape_info.items():
                anchor_pars = shape_type_info.pop("anchor_pars")
                shape_obj = eval(shape)(**shape_type_info)
                shape_container[f"{shape}.{shape_type}"] = shape_obj
                if anchor_pars != None:
                    shape_obj.make_anchors(*anchor_pars)
                else:
                    shape_obj.make_anchors()
        return shape_container

    @staticmethod
    def formate_callbacks(callbacks_info):
        # callbacks could be already a dict or a compact form that needs to be unpacked to make dict
        # case 1: {0: ['callback_1', 'common_arg', 'arg1']; 1: ['callback_1','common_arg', 'arg2']}
        # case 2: {'index': [0,1], 'callback': ['callback_1', 'common_arg'], 'args': ['arg1','arg2']}
        if ("index" in callbacks_info) and ("callback" in callbacks_info):
            callback = callbacks_info["callback"]
            if "args" in callbacks_info:
                callbacks = [callback + [arg] for arg in callbacks_info["args"]]
            else:
                callbacks = [callback] * len(callbacks_info["index"])
            return dict(zip(callbacks_info["index"], callbacks))
        else:
            return callbacks_info

    @classmethod
    def build_composite_shape_from_yaml(cls, yaml_file_path, **kwargs):
        with open(yaml_file_path, "r", encoding="utf8") as f:
            config = yaml.safe_load(f.read())
        shape_container = cls.build_basic_shape_from_yaml(yaml_file_path)
        composite_container = config["composite_shapes"]
        composite_obj_container = {}
        for i, (composite, composite_info) in enumerate(composite_container.items()):
            inherited = composite_info.pop("inherit", None)
            if inherited != None:
                inherited_composite_info = composite_container[inherited]
                composite_info = {**inherited_composite_info, **composite_info}
            _models = composite_info["models"]
            # _models could be a dict already or a compact way that will need to unpack to form a dict
            if ("model" in _models) and ("index" in _models):
                _models = dict(
                    zip(_models["index"], [_models["model"]] * len(_models["index"]))
                )
            hide_shape_ix = composite_info.pop("hide", [])
            callbacks_upon_model_change = buildTools.formate_callbacks(
                composite_info["callbacks_upon_model_change"]
            )
            callbacks_upon_leftmouse_click = buildTools.formate_callbacks(
                composite_info["callbacks_upon_leftmouse_click"]
            )
            callbacks_upon_rightmouse_click = buildTools.formate_callbacks(
                composite_info["callbacks_upon_rightmouse_click"]
            )
            callbacks = {
                "callbacks_upon_model_change": callbacks_upon_model_change,
                "callbacks_upon_leftmouse_click": callbacks_upon_leftmouse_click,
                "callbacks_upon_rightmouse_click": callbacks_upon_rightmouse_click,
            }
            shapes_tag = composite_info["shapes"]
            shapes = []
            for each in shapes_tag:
                if "*" not in each:
                    each = each + "*1"
                shape_key, num_shape = each.rsplit("*")
                num_shape = int(num_shape)
                for i in range(num_shape):
                    shapes.append(copy.deepcopy(shape_container[shape_key]))
            if "dynamic_shape_attributes" in composite_info:
                for shape_ix_tuple in composite_info["dynamic_shape_attributes"]:
                    attr_dict = composite_info["dynamic_shape_attributes"][
                        shape_ix_tuple
                    ]
                    for shape_ix_ in eval(shape_ix_tuple):
                        for attr_, value_ in attr_dict.items():
                            setattr(
                                shapes[shape_ix_], f"_dynamic_attribute_{attr_}", value_
                            )
            ref_shape = composite_info["ref_shape"]
            alignment_pattern = composite_info["alignment"]
            if "connection" in composite_info:
                connection_pattern = composite_info["connection"]
            else:
                connection_pattern = None
            static_labels = composite_info.pop("static_labels", [])
            composite_obj_container[composite] = shapeComposite(
                shapes=shapes,
                alignment_pattern=alignment_pattern,
                connection_pattern=connection_pattern,
                static_labels=static_labels,
                callbacks=callbacks,
                models=_models,
            )
            if "beam_height_offset" in composite_info:
                beamheight_offset = composite_info["beam_height_offset"]
            else:
                beamheight_offset = 0
            composite_obj_container[composite].set_beam_height_offset(beamheight_offset)
            if composite_info["transformation"] != "None":
                translate = composite_info["transformation"].pop("translate", (0, 0))
                if "translate" in kwargs:
                    assert (
                        type(kwargs["translate"][0]) == list
                        and len(kwargs["translate"]) > i
                    ), "You should provide a list of translate vector for all composite!"
                    translate = kwargs["translate"][i]
                rotation = composite_info["transformation"].pop("rotate", 0)
                sf = composite_info["transformation"].pop("scale", 1)
                composite_obj_container[composite].translate(translate)
                composite_obj_container[composite].rotate(rotation)
                composite_obj_container[composite].scale(sf)
            for i in hide_shape_ix:
                composite_obj_container[composite].shapes[i].hide_shape()
            composite_obj_container[composite].unpack_callbacks_and_models()

        return composite_obj_container

    @classmethod
    def build_view_from_yaml(
        cls,
        yaml_file_path,
        canvas_width,
        which_viewer=None,
        composite_obj_container=None,
    ):
        if composite_obj_container == None:
            composite_obj_container = cls.build_composite_shape_from_yaml(
                yaml_file_path
            )
        with open(yaml_file_path, "r", encoding="utf8") as f:
            viewer_config = yaml.safe_load(f.read())["viewers"]
        viewer_container = {}
        connection_container = {}
        if (which_viewer != None) and (which_viewer in viewer_config):
            viewer_config = {which_viewer: viewer_config[which_viewer]}
        for viewer, viewer_info in viewer_config.items():
            if viewer_info["transformation"]["translate"]["type"] == "absolute":
                max_width = 0
                max_width = max(
                    [
                        each[0]
                        for each in viewer_info["transformation"]["translate"]["values"]
                    ]
                )
                sf = canvas_width / max_width
            else:
                sf = 1

            def _build_view(viewer_info):
                composite_obj_container_subset = {}
                acc_boundary_offset_x = 0
                acc_boundary_offset_y = 0
                for i, each in enumerate(viewer_info["composites"]):
                    init_kwargs, cbs, models = composite_obj_container[
                        each
                    ].copy_object_meta()
                    init_kwargs.update({"callbacks": cbs, "models": models})
                    composite = shapeComposite(**init_kwargs)
                    composite.unpack_callbacks_and_models()
                    composite.set_shape_parent()
                    if i == 0:
                        translate = viewer_info["transformation"]["translate"][
                            "first_composite"
                        ]
                        translate = [int(translate[0] * sf), translate[1]]
                    else:
                        if (
                            viewer_info["transformation"]["translate"]["type"]
                            == "absolute"
                        ):
                            translate = viewer_info["transformation"]["translate"][
                                "values"
                            ][i - 1]
                            translate = [int(translate[0] * sf), translate[1]]
                        elif (
                            viewer_info["transformation"]["translate"]["type"]
                            == "relative"
                        ):
                            translate = (
                                viewer_info["transformation"]["translate"][
                                    "first_composite"
                                ]
                                + np.array(
                                    viewer_info["transformation"]["translate"][
                                        "values"
                                    ][0]
                                )
                                * i
                            ) + [acc_boundary_offset_x, acc_boundary_offset_y]
                    x_min, x_max, *_ = (
                        buildTools.calculate_boundary_for_combined_shapes(
                            composite.shapes
                        )
                    )
                    acc_boundary_offset_x = acc_boundary_offset_x + abs(x_max - x_min)
                    acc_boundary_offset_y = (
                        acc_boundary_offset_y + composite.beam_height_offset
                    )
                    composite.translate(translate)
                    if each in composite_obj_container_subset:
                        j = 2
                        while True:
                            if each + f"{j}" in composite_obj_container_subset:
                                j = j + 1
                                continue
                            else:
                                composite_obj_container_subset[each + f"{j}"] = (
                                    composite
                                )
                                break
                    else:
                        composite_obj_container_subset[each] = composite
                return composite_obj_container_subset

            if "cut_points" not in viewer_info:
                viewer_container[viewer] = _build_view(viewer_info)
            else:
                viewer_container[viewer] = {}
                inx_chain = (
                    [0] + viewer_info["cut_points"] + [len(viewer_info["composites"])]
                )
                inx_boundary = [
                    [inx_chain[i], inx_chain[i + 1]] for i in range(len(inx_chain) - 1)
                ]
                for i, [l, r] in enumerate(inx_boundary):
                    viewer_info_temp = {
                        "composites": viewer_info["composites"][l:r],
                        "transformation": {
                            "translate": {
                                "type": viewer_info["transformation"]["translate"][
                                    "type"
                                ][i],
                                "first_composite": viewer_info["transformation"][
                                    "translate"
                                ]["first_composite"][i],
                                "values": viewer_info["transformation"]["translate"][
                                    "values"
                                ][i],
                            }
                        },
                    }
                    viewer_container[viewer].update(_build_view(viewer_info_temp))
            if "connection" in viewer_info:
                connection_container[viewer] = viewer_info["connection"]
            else:
                connection_container[viewer] = {}
        return viewer_container, connection_container, composite_obj_container

    @classmethod
    def calculate_boundary_for_combined_shapes(cls, shapes):
        x_min, x_max, y_min, y_max = None, None, None, None
        for i, shape in enumerate(shapes):
            if i == 0:
                x_min, x_max, y_min, y_max = shape.calculate_shape_boundary()
            else:
                _x_min, _x_max, _y_min, _y_max = shape.calculate_shape_boundary()
                x_min = min([x_min, _x_min])
                x_max = max([x_max, _x_max])
                y_min = min([y_min, _y_min])
                y_max = max([y_max, _y_max])
        return x_min, x_max, y_min, y_max

    @classmethod
    def align_multiple_shapes(cls, shapes, orientations):
        def _align_shapes(_shapes, orientations_):
            # _shapes = [ref1, tag1, ref2, tag2, ...], orientations_ = [ref_or1, tag_or1, ref_or2, tag_or2, ...]
            assert len(_shapes) == len(
                orientations_
            ), "The length of shapes and orientation must be equal!"
            for i in range(len(_shapes) - 1):
                ref_shape, target_shape = _shapes[i], _shapes[i + 1]
                orientations_temp = orientations_[i : i + 2]
                buildTools.align_two_shapes(ref_shape, target_shape, orientations_temp)

        if type(shapes[0]) == list:
            # shapes is a list of _shapes, orientations is a list of orientations_
            assert (
                type(orientations[0]) == list
            ), "Format mismatch. Should be list of list."
            for shape_segment, orientaion_seg in zip(shapes, orientations):
                _align_shapes(shape_segment, orientaion_seg)
        else:
            _align_shapes(shapes, orientations)

    @classmethod
    def align_two_shapes(
        cls,
        ref_shape,
        target_shape,
        orientations=["bottom", "top"],
        gap=0.0,
        ref_anchors=[None, None],
        gaps_absolute=False,
    ):
        if orientations == ["translate", "translate"]:
            assert (
                type(gap) == list and len(gap) == 2
            ), "In translate mode, you should give two translation value in x and y directions"
            target_shape.transformation.update(ref_shape.transformation)
            if type(gap[0]) == str:
                assert (
                    gap[0] == "dx_for_fix_exit"
                ), "Wrong str, only dx_for_fix_exit accepted!"
                dx = getattr(target_shape.parent, gap[0])
                assert gap[1] == "yoffset", "Wrong str, only yoffset accepted!"
                dy = getattr(ref_shape, "_dynamic_attribute_" + gap[1])
            target_shape.transformation["translate"] = list(
                target_shape.transformation["translate"] + np.array(gap)
            )
            return target_shape
        cen_, v_unit = ref_shape.calculate_orientation_vector(
            orientations[0], ref_anchors[0]
        )
        v_mag = ref_shape.calculate_orientation_length(
            orientations[0], ref_anchors[0]
        ) + target_shape.calculate_orientation_length(orientations[1], ref_anchors[1])
        if gaps_absolute:
            v = v_unit * (v_mag + gap)
        else:
            v = v_unit * v_mag * (1 + gap)
        # set rot ang to 0 and translate to 0
        target_shape.reset()
        # this center is the geometry center if ref_anchor is None, and become offseted anchor otherwise
        if orientations[1] in ["left", "right", "top", "bottom"]:
            origin_cen_target = target_shape.compute_center_from_dim()
        else:
            if ref_anchors[1] == None:
                origin_cen_target = target_shape.compute_center_from_dim()
            else:
                if orientations[1] == "cen":
                    anchor = target_shape.compute_center_from_dim()
                else:
                    anchor = target_shape.anchors[orientations[1]]
                ref_anchor_offset = {
                    "left": np.array([-1, 0]),
                    "right": np.array([1, 0]),
                    "top": np.array([0, -1]),
                    "bottom": np.array([0, 1]),
                }
                assert ref_anchors[1] in ref_anchor_offset, "Wrong key for ref anchor"
                origin_cen_target = anchor - ref_anchor_offset[ref_anchors[1]]
        new_cen_target = v + cen_
        v_diff = new_cen_target - origin_cen_target
        target_shape.rot_center = origin_cen_target
        # let's calculate the angle between the original target shape and the orientated one
        target_cen_, target_v_unit = target_shape.calculate_orientation_vector(
            orientations[1], ref_anchors[1]
        )
        target_v_new = -v
        angle_offset = -angle_between(target_v_unit, target_v_new)
        target_shape.transformation.update(
            {"rotate": angle_offset, "translate": v_diff}
        )
        return target_shape

    @classmethod
    def make_line_connection_btw_two_anchors(
        cls, shapes, anchors, short_head_line_len=0, direct_connection=False, **kwargs
    ):
        line_nodes = []

        def _apply_offset(pos, dir):
            offset = {
                "left": np.array([-short_head_line_len, 0]),
                "right": np.array([short_head_line_len, 0]),
                "top": np.array([0, -short_head_line_len]),
                "bottom": np.array([0, short_head_line_len]),
            }
            return np.array(pos) + offset[dir]

        def _extend_to_beyond_boundary(pos, dir, pair_pos, overshot_pix_ct=20):
            x_min, x_max, y_min, y_max = (
                buildTools.calculate_boundary_for_combined_shapes(shapes)
            )
            x_pair, y_pair = pair_pos
            if dir == "left":
                if x_pair > pos[0]:
                    x = min([x_min, pos[0]]) - overshot_pix_ct
                    y = pos[1]
                else:
                    x, y = pos
            elif dir == "right":
                if x_pair < pos[0]:
                    x = max([x_max, pos[0]]) + overshot_pix_ct
                    y = pos[1]
                else:
                    x, y = pos
            elif dir == "top":
                if y_pair < pos[1]:
                    x = pos[0]
                    y = min([y_min, pos[1]]) - overshot_pix_ct
                else:
                    x, y = pos[0], min([pos[1], 100])
            elif dir == "bottom":
                if y_pair > pos[1]:
                    x = pos[0]
                    y = max([y_max, pos[1]]) + overshot_pix_ct
                else:
                    x, y = pos
            return [int(x), int(y)]

        def _get_sign_from_dir(dir):
            if dir in ["left", "top"]:
                return ">="
            elif dir in ["right", "bottom"]:
                return "<="

        assert (
            len(shapes) == 2 and len(anchors) == 2
        ), "shapes and anchors must be list of two items"
        dirs = []
        anchor_pos = []
        for shape, anchor in zip(shapes, anchors):
            if anchor == "dynamic_anchor":
                dirs.append("top")
                if (
                    shapes.index(shape) == 0
                ):  # if left anchor is dynamic, it is ready to use
                    if kwargs["side"][0] not in shape.dynamic_anchor:
                        return []
                    else:
                        anchor_pos.append(shape.dynamic_anchor[kwargs["side"][0]])
                else:
                    # print(kwargs['angle'])
                    anchor_pos.append(
                        shape.calculate_dynamic_anchor(
                            anchor_pos[0], kwargs["angle"], kwargs["side"][1]
                        )
                    )
                    if type(anchor_pos[-1]) == type(
                        None
                    ):  # no crosspoint due to parallel lines
                        return []
            else:
                dirs.append(shape.get_proper_extention_dir_for_one_anchor(anchor))
                anchor_pos.append(
                    shape.compute_anchor_pos_after_transformation(
                        anchor, return_pos_only=True
                    )
                )

        if direct_connection:
            line_pos = []
            for _pos, _dir in zip(anchor_pos, dirs):
                pos_offset = _apply_offset(_pos, _dir)
                if (_pos == anchor_pos[0]).all():
                    line_pos = line_pos + [_pos, pos_offset]
                else:
                    line_pos = line_pos + [pos_offset, _pos]
            return np.array(line_pos).astype(int)

        dir0, dir1 = dirs
        anchor_pos_offset = [
            _apply_offset(_pos, _dir) for _pos, _dir in zip(anchor_pos, dirs)
        ]

        if ("left" not in dirs) and ("right" not in dirs):
            if (dirs == ["top", "top"]) or (dirs == ["bottom", "bottom"]):
                first_anchor_pos_after_extend = _extend_to_beyond_boundary(
                    anchor_pos_offset[0], dir0, anchor_pos_offset[1]
                )
                second_anchor_pos_after_extend = _extend_to_beyond_boundary(
                    anchor_pos_offset[1], dir1, anchor_pos_offset[0]
                )
                if dirs == ["top", "top"]:
                    y_min = min(
                        [
                            first_anchor_pos_after_extend[1],
                            second_anchor_pos_after_extend[1],
                        ]
                    )
                else:
                    y_min = max(
                        [
                            first_anchor_pos_after_extend[1],
                            second_anchor_pos_after_extend[1],
                        ]
                    )
                first_anchor_pos_after_extend = [
                    first_anchor_pos_after_extend[0],
                    y_min,
                ]
                second_anchor_pos_after_extend = [
                    second_anchor_pos_after_extend[0],
                    y_min,
                ]
                line_nodes = [
                    anchor_pos[0],
                    anchor_pos_offset[0],
                    first_anchor_pos_after_extend,
                    second_anchor_pos_after_extend,
                    anchor_pos_offset[1],
                    anchor_pos[1],
                ]
            else:
                if (
                    (dir0 == "top")
                    and (anchor_pos_offset[0][1] < anchor_pos_offset[1][1])
                ) or (
                    (dir0 == "bottom")
                    and (anchor_pos_offset[0][1] > anchor_pos_offset[1][1])
                ):
                    first_anchor_pos_after_extend = _extend_to_beyond_boundary(
                        anchor_pos_offset[0], dir0, anchor_pos_offset[1]
                    )
                    second_anchor_pos_after_extend = _extend_to_beyond_boundary(
                        anchor_pos_offset[1], dir1, anchor_pos_offset[0]
                    )
                    x_cen = (anchor_pos_offset[0][0] + anchor_pos_offset[1][0]) / 2
                    first_anchor_pos_after_extend_cen = [
                        x_cen,
                        first_anchor_pos_after_extend[1],
                    ]
                    second_anchor_pos_after_extend_cen = [
                        x_cen,
                        second_anchor_pos_after_extend[1],
                    ]
                    line_nodes = [
                        anchor_pos[0],
                        anchor_pos_offset[0],
                        first_anchor_pos_after_extend,
                        first_anchor_pos_after_extend_cen,
                        second_anchor_pos_after_extend_cen,
                        second_anchor_pos_after_extend,
                        anchor_pos_offset[1],
                        anchor_pos[1],
                    ]
                else:
                    if anchor_pos_offset[0][1] < anchor_pos_offset[1][1]:
                        cross_pt = [anchor_pos_offset[1][0], anchor_pos_offset[0][1]]
                    else:
                        cross_pt = [anchor_pos_offset[0][0], anchor_pos_offset[1][1]]
                    line_nodes = [
                        anchor_pos[0],
                        anchor_pos_offset[0],
                        cross_pt,
                        anchor_pos_offset[1],
                        anchor_pos[1],
                    ]
        elif ("top" not in dirs) and ("bottom" not in dirs):
            if (dirs == ["left", "left"]) or (dirs == ["right", "right"]):
                first_anchor_pos_after_extend = _extend_to_beyond_boundary(
                    anchor_pos_offset[0], dir0, anchor_pos_offset[1]
                )
                second_anchor_pos_after_extend = _extend_to_beyond_boundary(
                    anchor_pos_offset[1], dir1, anchor_pos_offset[0]
                )
                if dirs == ["left", "left"]:
                    x_min = min(
                        [
                            first_anchor_pos_after_extend[0],
                            second_anchor_pos_after_extend[0],
                        ]
                    )
                else:
                    x_min = max(
                        [
                            first_anchor_pos_after_extend[0],
                            second_anchor_pos_after_extend[0],
                        ]
                    )
                first_anchor_pos_after_extend = [
                    x_min,
                    first_anchor_pos_after_extend[1],
                ]
                second_anchor_pos_after_extend = [
                    x_min,
                    second_anchor_pos_after_extend[1],
                ]
                line_nodes = [
                    anchor_pos[0],
                    anchor_pos_offset[0],
                    first_anchor_pos_after_extend,
                    second_anchor_pos_after_extend,
                    anchor_pos_offset[1],
                    anchor_pos[1],
                ]
            else:
                if (
                    (dir0 == "left")
                    and (anchor_pos_offset[0][0] < anchor_pos_offset[1][0])
                ) or (
                    (dir0 == "right")
                    and (anchor_pos_offset[0][0] > anchor_pos_offset[1][0])
                ):
                    first_anchor_pos_after_extend = _extend_to_beyond_boundary(
                        anchor_pos_offset[0], dir0, anchor_pos_offset[1]
                    )
                    second_anchor_pos_after_extend = _extend_to_beyond_boundary(
                        anchor_pos_offset[1], dir1, anchor_pos_offset[0]
                    )
                    y_cen = (anchor_pos_offset[0][1] + anchor_pos_offset[1][1]) / 2
                    first_anchor_pos_after_extend_cen = [
                        first_anchor_pos_after_extend[0],
                        y_cen,
                    ]
                    second_anchor_pos_after_extend_cen = [
                        second_anchor_pos_after_extend[0],
                        y_cen,
                    ]
                    line_nodes = [
                        anchor_pos[0],
                        anchor_pos_offset[0],
                        first_anchor_pos_after_extend,
                        first_anchor_pos_after_extend_cen,
                        second_anchor_pos_after_extend_cen,
                        second_anchor_pos_after_extend,
                        anchor_pos_offset[1],
                        anchor_pos[1],
                    ]
                else:
                    if anchor_pos_offset[0][1] < anchor_pos_offset[1][1]:
                        cross_pt = [anchor_pos_offset[1][0], anchor_pos_offset[0][1]]
                    else:
                        cross_pt = [anchor_pos_offset[0][0], anchor_pos_offset[1][1]]
                    line_nodes = [
                        anchor_pos[0],
                        anchor_pos_offset[0],
                        cross_pt,
                        anchor_pos_offset[1],
                        anchor_pos[1],
                    ]
        else:  # mixture of top/bottom and left/right
            if dir0 in ["top", "bottom"]:
                ref_x, ref_y = [anchor_pos_offset[0][0], anchor_pos_offset[1][1]]
                check_x, check_y = [anchor_pos_offset[1][0], anchor_pos_offset[0][1]]
                check_result_x = eval(f"{check_x}{_get_sign_from_dir(dir1)}{ref_x}")
                check_result_y = eval(f"{check_y}{_get_sign_from_dir(dir0)}{ref_y}")
                if check_result_x and check_result_y:
                    cross_pt = [ref_x, ref_y]
                    line_nodes = [
                        anchor_pos[0],
                        anchor_pos_offset[0],
                        cross_pt,
                        anchor_pos_offset[1],
                        anchor_pos[1],
                    ]
                else:
                    first_anchor_pos_after_extend = _extend_to_beyond_boundary(
                        anchor_pos_offset[0], dir0, anchor_pos_offset[1]
                    )
                    second_anchor_pos_after_extend = _extend_to_beyond_boundary(
                        anchor_pos_offset[1], dir1, anchor_pos_offset[0]
                    )
                    cross_pt = [
                        second_anchor_pos_after_extend[0],
                        first_anchor_pos_after_extend[1],
                    ]
                    line_nodes = [
                        anchor_pos[0],
                        anchor_pos_offset[0],
                        first_anchor_pos_after_extend,
                        cross_pt,
                        second_anchor_pos_after_extend,
                        anchor_pos_offset[1],
                        anchor_pos[1],
                    ]
            else:
                ref_x, ref_y = [anchor_pos_offset[1][0], anchor_pos_offset[0][1]]
                check_x, check_y = [anchor_pos_offset[0][0], anchor_pos_offset[1][1]]
                check_result_x = eval(f"{check_x}{_get_sign_from_dir(dir0)}{ref_x}")
                check_result_y = eval(f"{check_y}{_get_sign_from_dir(dir1)}{ref_y}")
                if check_result_x and check_result_y:
                    cross_pt = [ref_x, ref_y]
                    line_nodes = [
                        anchor_pos[0],
                        anchor_pos_offset[0],
                        cross_pt,
                        anchor_pos_offset[1],
                        anchor_pos[1],
                    ]
                else:
                    first_anchor_pos_after_extend = _extend_to_beyond_boundary(
                        anchor_pos_offset[0], dir0, anchor_pos_offset[1]
                    )
                    second_anchor_pos_after_extend = _extend_to_beyond_boundary(
                        anchor_pos_offset[1], dir1, anchor_pos_offset[0]
                    )
                    cross_pt = [
                        first_anchor_pos_after_extend[0],
                        second_anchor_pos_after_extend[1],
                    ]
                    line_nodes = [
                        anchor_pos[0],
                        anchor_pos_offset[0],
                        first_anchor_pos_after_extend,
                        cross_pt,
                        second_anchor_pos_after_extend,
                        anchor_pos_offset[1],
                        anchor_pos[1],
                    ]
        return np.array(line_nodes).astype(int)
