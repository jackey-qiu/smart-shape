from PyQt5.QtGui import (
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
from ...util.geometry_transformation import rotate_multiple_points, angle_between, line_intersection

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


class baseShape(object):

    def __init__(
        self,
        dim,
        decoration_cursor_off=DECORATION_UPON_CURSOR_OFF,
        decoration_cursor_on=DECORATION_UPON_CURSOR_ON,
        rotation_center=None,
        transformation={"rotate": 0, "translate": (0, 0), 'translate_offset':[0,0], "scale": 1},
        text_decoration=DECORATION_TEXT_DEFAULT,
        lables={"text": [], "anchor": [], "orientation": [], "decoration": None},
    ):
        self._dim_pars = dim
        self._dim_pars_origin = dim
        self.ref_geometry = transformation["translate"]

        self.anchor_kwargs = None
        self._rotcenter = rotation_center
        self._decoration = copy.deepcopy(decoration_cursor_off)
        self._decoration_cursor_on = copy.deepcopy(decoration_cursor_on)
        self._decoration_cursor_off = copy.deepcopy(decoration_cursor_off)
        self.anchors = {}
        self.dynamic_anchor = {}
        self.ang_of_incoming_line = 0
        self._transformation = transformation
        self._text_decoration = copy.deepcopy(text_decoration)
        self._labels = copy.deepcopy(lables)
        self.clickable = False
        self.show = True
        self.parent = None

    def set_parent(self, parent):
        #set the parent of this shape, usually the composit object that contain this shape
        self.parent = parent

    def set_clickable(self, clickable=True):
        self.clickable = clickable

    def reset_ref_geometry(self):
        self.ref_geometry = copy.deepcopy(self.transformation["translate"])

    def show_shape(self):
        self.show = True

    def hide_shape(self):
        self.show = False

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        assert type(labels) == dict, "Need dictionary for labels"
        assert "text" in labels, "need text at least"
        if len(labels["text"]) != 0:
            self._labels.update(labels)

    @property
    def text_decoration(self):
        return self._text_decoration

    @text_decoration.setter
    def text_decoration(self, decoration):
        self._text_decoration.update(decoration)

    @property
    def dim_pars(self):
        return self._dim_pars

    @dim_pars.setter
    def dim_pars(self, new_dim):
        self._dim_pars = new_dim
        self._dim_pars_origin = new_dim

    @property
    def rot_center(self):
        if self._rotcenter == None:
            self._rotcenter = [
                int(each) for each in self.compute_center_from_dim(False)
            ]
        return self._rotcenter

    @rot_center.setter
    def rot_center(self, rot_center):
        if (
            type(rot_center) == tuple
            or type(rot_center) == list
            or type(rot_center) == np.ndarray
        ):
            self._rotcenter = [int(each) for each in rot_center]
        elif rot_center == None:
            self._rotcenter = [
                int(each) for each in self.compute_center_from_dim(False)
            ]

    @property
    def decoration(self):
        return self._decoration

    @decoration.setter
    def decoration(self, decoration):
        self._decoration.update(decoration)

    @property
    def decoration_cursor_on(self):
        return self._decoration_cursor_on

    @decoration_cursor_on.setter
    def decoration_cursor_on(self, decoration):
        self._decoration_cursor_on.update(decoration)

    @property
    def decoration_cursor_off(self):
        return self._decoration_cursor_off

    @decoration_cursor_off.setter
    def decoration_cursor_off(self, decoration):
        self._decoration_cursor_off.update(decoration)

    @property
    def transformation(self):
        return self._transformation

    @transformation.setter
    def transformation(self, transformation):
        assert isinstance(transformation, dict), "wrong format of transformation"
        self._transformation.update(transformation)

    def compute_center_from_dim(self, apply_translate=True):
        raise NotImplementedError

    def make_anchors(self, **kwargs):
        self.anchor_kwargs = kwargs

    def calculate_dynamic_anchor(self, ref_pt, angle, which_side):
        ang = math.radians(angle)
        dx = math.cos(ang)
        dy = -math.sin(ang) #in viewport, y axis goes from low to heigher pointing downwards
        ref_pt_2 = np.array(ref_pt) + [dx, dy]
        res = self.calculate_crosspoint([ref_pt, ref_pt_2], which_side)
        if res!=None:
            self.dynamic_anchor.update({which_side: np.array(res).astype(int)})
            return self.dynamic_anchor[which_side]

    def update_anchors(self):
        self.make_anchors(**self.anchor_kwargs)

    def update_ang_incoming_line(self, ang):
        self.ang_of_incoming_line = ang

    def calculate_shape(self):
        raise NotImplementedError

    def calculate_corners(self):
        raise NotImplementedError
    
    def calculate_crosspoint(self, two_pts_on_line, which_side):
        raise NotImplementedError

    def calculate_anchor_orientation(self):
        raise NotImplementedError

    def calculate_shape_boundary(self):
        raise NotImplementedError

    def check_pos(self, x, y):
        raise NotImplementedError

    def text_label(self, qp):
        raise NotImplementedError

    def _draw_text(
        self,
        qp,
        alignment,
        text,
        anchor,
        x,
        y,
        width,
        height,
        width_txt,
        height_txt,
        padding,
        txt_orientation="horizontal",
    ):
        # qp: qpainter
        # alignment: Qt alignment enum
        # text: text label
        # (x, y) original anchor position
        # width, height: the width and height of shape
        # width_txt, height_txt: the width and height of text
        # padding: additinal padding to applied
        # net effect: the text will be displayed at the anchor position after considering the final size of text area and orientation sense
        padding = 0
        if anchor == "left":
            x = x - width_txt - padding
            y = y + int((height - height_txt) / 2)
        elif anchor == "right":
            x = x + width + padding
            y = y + int((height - height_txt) / 2)
        elif anchor == "top":
            y = y - height_txt - padding
            x = x + int((width - width_txt) / 2)
        elif anchor == "bottom":
            y = y + height + padding
            x = x + int((width - width_txt) / 2)
        elif anchor == "center":
            x = x + int((width - width_txt) / 2) + padding
            y = y + int((height - height_txt) / 2) + padding
        else:
            if anchor in self.anchors:
                x, y = self.anchors[anchor]
                if "left" in anchor:
                    x = x - width_txt - padding
                    y = y + int((0 - height_txt) / 2)
                elif "right" in anchor:
                    x = x + 0 + padding
                    y = y + int((0 - height_txt) / 2)
                elif "top" in anchor:
                    y = y - height_txt - padding
                    x = x + int((0 - width_txt) / 2)
                elif "bottom" in anchor:
                    y = y + 0 + padding
                    x = x + int((0 - width_txt) / 2)
            else:
                raise KeyError("Invalid anchor key for text labeling!")
        if txt_orientation == "horizontal":
            if "top" in anchor:
                qp.drawText(
                    int(x),
                    int(y - height_txt / 2),
                    int(width_txt),
                    int(height_txt),
                    getattr(Qt, alignment),
                    text,
                )
            elif "bottom" in anchor:
                qp.drawText(
                    int(x),
                    int(y + height_txt / 2),
                    int(width_txt),
                    int(height_txt),
                    getattr(Qt, alignment),
                    text,
                )
            elif "center" in anchor:
                qp.drawText(
                    int(x),
                    int(y),
                    int(width_txt),
                    int(height_txt),
                    getattr(Qt, alignment),
                    text,
                )
            elif "left" in anchor:
                qp.drawText(
                    int(x - height_txt / 2),
                    int(y),
                    int(width_txt),
                    int(height_txt),
                    getattr(Qt, alignment),
                    text,
                )
            elif "right" in anchor:
                qp.drawText(
                    int(x + height_txt / 2),
                    int(y),
                    int(width_txt),
                    int(height_txt),
                    getattr(Qt, alignment),
                    text,
                )
            else:
                qp.drawText(
                    int(x),
                    int(y),
                    int(width_txt),
                    int(height_txt),
                    getattr(Qt, alignment),
                    text,
                )
        elif txt_orientation == "vertical":
            if "right" in anchor:
                qp.translate(
                    int(x + height_txt / 2), int(y + width_txt / 2 + height_txt / 2)
                )
            elif "left" in anchor:
                qp.translate(
                    int(x + width_txt - height_txt * 1.5),
                    int(y + width_txt / 2 + height_txt / 2),
                )
            elif "top" in anchor:
                qp.translate(
                    int(x + width_txt / 2 - height_txt / 2), int(y + height_txt / 2)
                )
            elif "bottom" in anchor:
                qp.translate(
                    int(x + width_txt / 2 - height_txt / 2),
                    int(y + width_txt + height_txt / 2),
                )
            elif "center" in anchor:
                qp.translate(
                    int(x + width_txt / 2 - height_txt / 2),
                    int(y + width_txt / 2 + height_txt / 2),
                )
            else:
                qp.drawText(
                    int(x),
                    int(y),
                    int(width_txt),
                    int(height_txt),
                    getattr(Qt, alignment),
                    text,
                )                
            qp.rotate(270)
            qp.drawText(
                int(0),
                int(0),
                int(width_txt),
                int(height_txt),
                getattr(Qt, alignment),
                text,
            )

    def get_proper_extention_dir_for_one_anchor(self, key):
        possible_dirs = []
        possible_dirs_offset = []
        anchor_pos, cen, _ = self.compute_anchor_pos_after_transformation(
            key, return_pos_only=False
        )
        orientations = {
            "left": np.array([-1, 0]),
            "right": np.array([1, 0]),
            "top": np.array([0, -1]),
            "bottom": np.array([0, 1]),
        }
        for each, value in orientations.items():
            if not self.check_pos(*(np.array(anchor_pos) + value)):
                possible_dirs.append(each)
                possible_dirs_offset.append(value)
        if len(possible_dirs) == 0:
            return None
        else:
            ix_shortest = np.argmin(
                np.linalg.norm(
                    cen - (np.array(anchor_pos) + np.array(possible_dirs_offset))
                )
            )
            return possible_dirs[ix_shortest]

    def compute_anchor_pos_after_transformation(
        self, key, return_pos_only=False, ref_anchor=None
    ):
        # calculate anchor pos for key after transformation
        # ref_anchor in [None, 'left', 'right', 'top', 'bottom']
        if ref_anchor == "None":
            ref_anchor = None
        ref_anchor_offset = {
            "left": np.array([-1, 0]),
            "right": np.array([1, 0]),
            "top": np.array([0, -1]),
            "bottom": np.array([0, 1]),
        }
        ref_anchor_dir = None
        if ref_anchor != None:
            ref_anchor_dir = ref_anchor_offset[ref_anchor]

        orientation = key
        or_len = self.calculate_orientation_length(orientation, ref_anchor=ref_anchor)
        cen = self.compute_center_from_dim(apply_translate=True)
        rotate_angle = (
            self.transformation["rotate"] if "rotate" in self.transformation else 0
        )
        anchor = None
        if orientation == "top":
            anchor = np.array(cen) + [0, -or_len]
            ref_anchor = cen
        elif orientation == "bottom":
            anchor = np.array(cen) + [0, or_len]
            ref_anchor = cen
        elif orientation == "left":
            anchor = np.array(cen) + [-or_len, 0]
            ref_anchor = cen
        elif orientation == "right":
            anchor = np.array(cen) + [or_len, 0]
            ref_anchor = cen
        elif orientation == "cen":
            anchor = np.array(cen)
            if ref_anchor != None:
                # ref_anchor = anchor - ref_anchor_offset[ref_anchor_dir]
                ref_anchor = anchor - ref_anchor_dir
            else:
                ref_anchor = anchor - ref_anchor_offset["left"]  # by default
        else:
            if orientation in self.anchors:
                anchor = np.array(self.anchors[orientation]) + np.array(
                    self.transformation["translate"]) + self.transformation['translate_offset']
                if ref_anchor != None:
                    # ref_anchor = anchor - ref_anchor_offset[ref_anchor_dir]
                    ref_anchor = anchor - ref_anchor_dir
                else:
                    ref_anchor = cen
            else:
                raise KeyError("Not the right key for orientation")
        rot_center = np.array(self.rot_center) + np.array(
            self.transformation["translate"]) + self.transformation['translate_offset']
        cen_, anchor_, ref_anchor_ = rotate_multiple_points(
            [cen, anchor, ref_anchor], rot_center, rotate_angle
        )
        # return cen and anchor pos and or_len after transformation
        if return_pos_only:
            return anchor_
        else:
            return anchor_, ref_anchor_, or_len

    def cursor_pos_checker(self, x, y):
        if not self.clickable:
            return False
        cursor_inside_shape = self.check_pos(x, y)
        if cursor_inside_shape:
            self.decoration = copy.deepcopy(self.decoration_cursor_on)
            return True
        else:
            self.decoration = copy.deepcopy(self.decoration_cursor_off)
            return False

    def apply_transform(self, qp):
        rotate_angle = (
            self.transformation["rotate"] if "rotate" in self.transformation else 0
        )
        rot_center = self.rot_center
        qp.translate(*rot_center)
        qp.translate(*(np.array(self.transformation["translate"])+self.transformation['translate_offset']))
        qp.rotate(rotate_angle)
        qp.translate(*[-each for each in rot_center])
        return qp

    def paint(self, qp) -> None:

        decoration = make_decoration_from_text(self.decoration)
        qp.setPen(decoration["pen"])
        qp.setBrush(decoration["brush"])
        self.draw_shape(qp)

    def draw_shape(self, qp):
        raise NotImplementedError

    def calculate_orientation_vector(self, orientation="top", ref_anchor=None):
        anchor_, cen_, or_len = self.compute_anchor_pos_after_transformation(
            orientation, return_pos_only=False, ref_anchor=ref_anchor
        )
        return cen_, (anchor_ - cen_) / or_len

    def translate(self, v):
        self.transformation = {"translate": v}

    def rotate(self, angle):
        self.transformation = {"rotate": angle}

    def scale(self, sf):
        raise NotImplementedError

    def reset(self):
        self.transformation.update({"rotate": 0, "translate": [0, 0]})


class rectangle(baseShape):
    def __init__(
        self,
        dim=[700, 100, 40, 80],
        rotation_center=None,
        decoration_cursor_off=DECORATION_UPON_CURSOR_OFF,
        decoration_cursor_on=DECORATION_UPON_CURSOR_ON,
        transformation={"rotate": 0, "translate": (0, 0), 'translate_offset':[0,0], "scale": 1},
        text_decoration=DECORATION_TEXT_DEFAULT,
        labels={"text": [], "anchor": [], "orientation": [], "decoration": None},
    ):

        super().__init__(
            dim=dim,
            rotation_center=rotation_center,
            decoration_cursor_off=decoration_cursor_off,
            decoration_cursor_on=decoration_cursor_on,
            transformation=transformation,
            text_decoration=text_decoration,
            lables=labels,
        )

    def scale(self, sf):
        self.dim_pars = (
            np.array(self.dim_pars)
            * [
                1,
                1,
                sf / self.transformation["scale"],
                sf / self.transformation["scale"],
            ]
        ).astype(int)
        self.transformation = {"scale": sf}
        if self.anchor_kwargs != None:
            self.update_anchors()

    def draw_shape(self, qp):
        qp = self.apply_transform(qp)
        if self.anchor_kwargs != None:
            self.update_anchors()
        if self.show:
            qp.drawRect(*np.array(self.dim_pars).astype(int))
        self.text_label(qp)

    def calculate_corners(self):
        x, y, w, h = self.dim_pars 
        p1 = [x, y]#top left
        p2 = [x+w,y]#top right
        p3 = [x, y+h]#bottom left
        p4 = [x+w, y+h]#bottom right
        pos_ = rotate_multiple_points(
                [each + np.array(self.transformation["translate"])+self.transformation['translate_offset'] for each in [p1,p2,p3,p4]],
                np.array(self.rot_center) + np.array(self.transformation["translate"])+self.transformation['translate_offset'],
                self.transformation["rotate"],
            )
        return pos_
    
    def calculate_crosspoint(self, two_pts_on_line, which_side):
        p_tl, p_tr, p_bl, p_br = self.calculate_corners()
        if which_side == 'top':
            target_line = [p_tl, p_tr]
        elif which_side == 'bottom':
            target_line = [p_bl, p_br]
        elif which_side == 'left':
            target_line = [p_tl, p_bl]
        elif which_side == 'right':
            target_line = [p_tl, p_br]
        else:
            print('which_side= {which_side} is not valid! Should be one of :top, bottom, left, right')
            return None
        crosspt = line_intersection(two_pts_on_line, target_line)
        return crosspt

    def text_label(self, qp):
        labels = self.labels
        decoration = self.text_decoration
        qp.save()
        for i, text in enumerate(labels["text"]):
            x, y, w, h = self.dim_pars
            anchor = labels["anchor"][i]
            if labels["decoration"] == None:
                decoration = self.text_decoration
            else:
                if type(labels["decoration"]) == list and len(
                    labels["decoration"]
                ) == len(labels["text"]):
                    decoration = labels["decoration"][i]
                else:
                    decoration = self.text_decoration
            alignment = decoration["alignment"]
            padding = decoration["padding"]
            text_color = decoration["text_color"]
            font_size = decoration["font_size"]
            qp.setPen(QColor(*text_color))
            qp.setFont(QFont("Decorative", font_size))
            text_bound_rect = qp.fontMetrics().boundingRect(
                QRect(), Qt.AlignCenter, text
            )
            w_txt, h_txt = text_bound_rect.width(), text_bound_rect.height()
            self._draw_text(
                qp,
                alignment,
                text,
                anchor,
                x,
                y,
                w,
                h,
                w_txt,
                h_txt,
                padding,
                labels["orientation"][i],
            )
            qp.restore()
            qp.save()

    def calculate_shape_boundary(self):
        x, y, w, h = self.dim_pars
        four_corners = [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]
        four_corners = [
            np.array(each) + np.array(self.transformation["translate"])
            for each in four_corners
        ]
        rot_center = np.array(self.rot_center) + np.array(
            self.transformation["translate"]
        )
        four_corners = rotate_multiple_points(
            four_corners, rot_center, self.transformation["rotate"]
        )
        return (
            int(four_corners[:, 0].min()),
            int(four_corners[:, 0].max()),
            int(four_corners[:, 1].min()),
            int(four_corners[:, 1].max()),
        )

    def compute_center_from_dim(self, apply_translate=True):
        x, y, w, h = self.dim_pars
        if apply_translate:
            return (
                x + w / 2 + self.transformation["translate"][0] + self.transformation['translate_offset'][0],
                y + h / 2 + self.transformation["translate"][1] + self.transformation['translate_offset'][1],
            )
        else:
            return x + w / 2, y + h / 2

    def make_anchors(
        self, num_of_anchors_on_each_side=5, include_corner=True, grid=False
    ):
        # num_of_anchors_on_each_side: exclude corners
        # two possible signature for num_of_achors_one_each_side
        # either a single int meaning same number of anchor points on top/bottom and left/right side
        # or a tuple of two int values meaning different number of anchor points
        # (10, 5): means 10 anchor points top/bottom side and 5 on left/right side
        # if grid is True, anchors will be not only on four edges but also inside the rectangle in a grid net
        super().make_anchors(
            num_of_anchors_on_each_side=num_of_anchors_on_each_side,
            include_corner=include_corner,
            grid=grid,
        )
        if type(num_of_anchors_on_each_side) == str:
            try:
                num_of_anchors_on_each_side = [int(num_of_anchors_on_each_side)] * 2
            except:
                num_of_anchors_on_each_side = eval(num_of_anchors_on_each_side)
        else:
            try:
                num_of_anchors_on_each_side = [int(num_of_anchors_on_each_side)] * 2
            except:
                num_of_anchors_on_each_side = num_of_anchors_on_each_side
        assert (
            len(num_of_anchors_on_each_side) == 2
        ), "You need two integer number to represent anchor number on all sides"

        w, h = self.dim_pars[2:]
        if not include_corner:
            w_step, h_step = w / (num_of_anchors_on_each_side[0] + 1), h / (
                num_of_anchors_on_each_side[1] + 1
            )
        else:
            assert (
                num_of_anchors_on_each_side[0] > 2
                and num_of_anchors_on_each_side[1] > 2
            ), "At least two achors at each edge"
            w_step, h_step = w / (num_of_anchors_on_each_side[0] - 1), h / (
                num_of_anchors_on_each_side[1] - 1
            )

        top_left_coord = np.array(self.dim_pars[0:2])
        bottom_right_coord = top_left_coord + np.array([w, h])
        anchors = {}
        if not grid:
            for i in range(num_of_anchors_on_each_side[0]):
                if not include_corner:
                    anchors[f"anchor_top_{i}"] = top_left_coord + [(i + 1) * w_step, 0]
                    anchors[f"anchor_bottom_{i}"] = bottom_right_coord + [
                        -(i + 1) * w_step,
                        0,
                    ]
                else:
                    anchors[f"anchor_top_{i}"] = top_left_coord + [i * w_step, 0]
                    anchors[f"anchor_bottom_{i}"] = bottom_right_coord + [
                        -i * w_step,
                        0,
                    ]
            for i in range(num_of_anchors_on_each_side[1]):
                if not include_corner:
                    anchors[f"anchor_left_{i}"] = top_left_coord + [0, (i + 1) * h_step]
                    anchors[f"anchor_right_{i}"] = bottom_right_coord + [
                        0,
                        -(i + 1) * h_step,
                    ]
                else:
                    anchors[f"anchor_left_{i}"] = top_left_coord + [0, i * h_step]
                    anchors[f"anchor_right_{i}"] = bottom_right_coord + [0, -i * h_step]
        else:
            for i in range(num_of_anchors_on_each_side[0]):  # num of columns
                for j in range(num_of_anchors_on_each_side[1]):  # num of rows
                    if not include_corner:
                        anchors[f"anchor_grid_{j}_{i}"] = top_left_coord + [
                            (i + 1) * w_step,
                            (j + 1) * h_step,
                        ]
                    else:
                        anchors[f"anchor_grid_{j}_{i}"] = top_left_coord + [
                            i * w_step,
                            j * h_step,
                        ]
        self.anchors = anchors

    def calculate_orientation_length(self, orientation="top", ref_anchor=None):
        if orientation == "cen":
            return 1
        w, h = np.array(self.dim_pars[2:])
        if orientation in ["top", "bottom"]:
            return h / 2
        elif orientation in ["left", "right"]:
            return w / 2
        else:
            if orientation in self.anchors:
                if ref_anchor == None:
                    return np.linalg.norm(
                        np.array(self.anchors[orientation])
                        - np.array(self.compute_center_from_dim(apply_translate=False))
                    )
                else:
                    return 1
            else:
                raise KeyError(
                    f"No such orientation key:{orientation}!Possible ones are {self.anchors}"
                )

    def check_pos(self, x, y):
        ox, oy, w, h = np.array(self.dim_pars)
        pos_ = rotate_multiple_points(
            [(x, y)],
            np.array(self.rot_center) + np.array(self.transformation["translate"]) + np.array(self.transformation["translate_offset"]),
            -self.transformation["rotate"],
        )
        pos_ = np.array(pos_) - np.array(self.transformation["translate"]) - np.array(self.transformation["translate_offset"])
        x_, y_ = pos_
        if (ox <= x_ <= ox + w) and (oy <= y_ <= oy + h):
            return True
        else:
            return False


class roundedRectangle(rectangle):
    def __init__(
        self,
        dim=[700, 100, 40, 80, 10, 10],
        rotation_center=None,
        decoration_cursor_off=DECORATION_UPON_CURSOR_OFF,
        decoration_cursor_on=DECORATION_UPON_CURSOR_ON,
        transformation={"rotate": 0, "translate": (0, 0), 'translate_offset':[0,0], "scale": 1},
        text_decoration=DECORATION_TEXT_DEFAULT,
        labels={"text": [], "anchor": [], "orientation": [], "decoration": None},
    ):
        super().__init__(
            dim[0:4],
            rotation_center,
            decoration_cursor_off,
            decoration_cursor_on,
            transformation,
            text_decoration,
            labels,
        )
        self.xy_radius = dim[4:]

    def draw_shape(self, qp):
        qp = self.apply_transform(qp)
        if self.anchor_kwargs != None:
            self.update_anchors()
        if self.show:
            qp.drawRoundedRect(
                *np.array(list(self.dim_pars) + list(self.xy_radius)).astype(int)
            )
        self.text_label(qp)


class circle(baseShape):
    def __init__(
        self,
        dim=[100, 100, 40],
        rotation_center=None,
        decoration_cursor_off=DECORATION_UPON_CURSOR_OFF,
        decoration_cursor_on=DECORATION_UPON_CURSOR_ON,
        transformation={"rotate": 0, "translate": (0, 0), 'translate_offset':[0,0], "scale": 1},
        text_decoration=DECORATION_TEXT_DEFAULT,
        labels={"text": [], "anchor": [], "orientation": [], "decoration": None},
    ):

        super().__init__(
            dim=dim,
            rotation_center=rotation_center,
            decoration_cursor_off=decoration_cursor_off,
            decoration_cursor_on=decoration_cursor_on,
            transformation=transformation,
            text_decoration=text_decoration,
            lables=labels,
        )

    def scale(self, sf):
        self.dim_pars = list(
            (
                np.array(self.dim_pars)
                * np.array([1, 1, sf / self.transformation["scale"]])
            ).astype(int)
        )
        self.transformation["scale"] = sf
        if self.anchor_kwargs != None:
            self.update_anchors()

    def draw_shape(self, qp):
        if self.anchor_kwargs != None:
            self.update_anchors()
        qp = self.apply_transform(qp)
        if self.show:
            qp.drawEllipse(*(self.dim_pars + [self.dim_pars[-1]]))
        self.text_label(qp)

    def text_label(self, qp):
        labels = self.labels
        decoration = self.text_decoration
        cen = self.compute_center_from_dim(False)
        r = self.dim_pars[-1] / 2
        qp.save()
        for i, text in enumerate(labels["text"]):
            x, y = cen
            x, y = x - r, y - r
            anchor = labels["anchor"][i]
            if labels["decoration"] == None:
                decoration = self.text_decoration
            else:
                if type(labels["decoration"]) == list and len(
                    labels["decoration"]
                ) == len(labels["text"]):
                    decoration = labels["decoration"][i]
                else:
                    decoration = self.text_decoration
            alignment = decoration["alignment"]
            padding = decoration["padding"]
            text_color = decoration["text_color"]
            font_size = decoration["font_size"]
            qp.setPen(QColor(*text_color))
            qp.setFont(QFont("Decorative", font_size))
            text_bound_rect = qp.fontMetrics().boundingRect(
                QRect(), Qt.AlignCenter, text
            )
            w_txt, h_txt = text_bound_rect.width(), text_bound_rect.height()
            self._draw_text(
                qp,
                alignment,
                text,
                anchor,
                x,
                y,
                2 * r,
                2 * r,
                w_txt,
                h_txt,
                padding,
                labels["orientation"][i],
            )
            qp.restore()
            qp.save()

    def calculate_shape_boundary(self):
        cen = np.array(self.compute_center_from_dim(False))
        r = self.dim_pars[-1] / 2
        four_corners = [cen + each for each in [[r, 0], [-r, 0], [0, r], [0, -r]]]
        four_corners = [
            np.array(each) + np.array(self.transformation["translate"])
            for each in four_corners
        ]
        rot_center = np.array(self.rot_center) + np.array(
            self.transformation["translate"]
        )
        four_corners = rotate_multiple_points(
            four_corners, rot_center, self.transformation["rotate"]
        )
        return (
            int(four_corners[:, 0].min()),
            int(four_corners[:, 0].max()),
            int(four_corners[:, 1].min()),
            int(four_corners[:, 1].max()),
        )

    def compute_center_from_dim(self, apply_translate=True):
        x, y, R = self.dim_pars
        x, y = x + R / 2, y + R / 2
        if apply_translate:
            return (
                x + self.transformation["translate"][0] + self.transformation['translate_offset'][0],
                y + self.transformation["translate"][1] + self.transformation['translate_offset'][1],
            )
        else:
            return x, y

    def make_anchors(self, num_of_anchors=4):
        # num_of_anchors_on_each_side: exclude corners
        *_, R = self.dim_pars
        r = R / 2
        super().make_anchors(num_of_anchors=num_of_anchors)
        cen = np.array(self.compute_center_from_dim(False))
        ang_step = math.radians(360 / num_of_anchors)
        anchors = {}
        for i in range(num_of_anchors):
            dx, dy = r * math.cos(ang_step * i), -r * math.sin(ang_step * i)
            anchors[f"anchor_{i}"] = cen + [dx, dy]
        self.anchors = anchors

    def calculate_orientation_length(self, orientation="top", ref_anchor=None):
        if orientation == "cen":
            return 1
        else:
            return self.dim_pars[-1] / 2

    def check_pos(self, x, y):
        cen = np.array(self.compute_center_from_dim(False))
        r = self.dim_pars[-1] / 2
        p1, p2, p3, p4 = [cen + each for each in [[r, 0], [-r, 0], [0, r], [0, -r]]]
        pos_ = rotate_multiple_points(
            [(x, y)],
            np.array(self.rot_center) + self.transformation["translate"] + self.transformation["translate_offset"],
            -self.transformation["rotate"],
        )
        pos_ = np.array(pos_) - (np.array(self.transformation["translate"]) + self.transformation["translate_offset"])
        x_, y_ = pos_
        if (p2[0] <= x_ <= p1[0]) and (p4[1] <= y_ <= p3[1]):
            return True
        else:
            return False

class isocelesTriangle(baseShape):
    def __init__(
        self,
        dim=[100, 100, 40, 60],
        rotation_center=None,
        decoration_cursor_off=DECORATION_UPON_CURSOR_OFF,
        decoration_cursor_on=DECORATION_UPON_CURSOR_ON,
        transformation={"rotate": 0, "translate": (0, 0), 'translate_offset':[0,0], "scale": 1},
        text_decoration=DECORATION_TEXT_DEFAULT,
        labels={"text": [], "anchor": [], "orientation": [], "decoration": None},
    ):

        super().__init__(
            dim=dim,
            rotation_center=rotation_center,
            decoration_cursor_off=decoration_cursor_off,
            decoration_cursor_on=decoration_cursor_on,
            transformation=transformation,
            text_decoration=text_decoration,
            lables=labels,
        )

    def scale(self, sf):
        self.dim_pars = (
            np.array(self.dim_pars) * [1, 1, sf / self.transformation["scale"], 1]
        ).astype(int)
        self.transformation["scale"] = sf
        if self.anchor_kwargs != None:
            self.update_anchors()

    def _cal_width_height(self):
        p1,p2,p3 = self._cal_corner_point_coordinates(False)
        return abs(p2[0]-p3[0]), abs(p1[1]-p2[1])

    def _cal_corner_point_coordinates(self, return_type_is_qpointF=True):
        ang = math.radians(self.dim_pars[-1]) / 2
        edge_lenth = self.dim_pars[-2]
        dx = edge_lenth * math.sin(ang)
        dy = edge_lenth * math.cos(ang)
        point1 = (np.array(self.dim_pars[0:2])).astype(int)
        point2 = (np.array(self.dim_pars[0:2]) + np.array([-dx, dy])).astype(int)
        point3 = (np.array(self.dim_pars[0:2]) + np.array([dx, dy])).astype(int)
        if return_type_is_qpointF:
            return QPointF(*point1), QPointF(*point2), QPointF(*point3)
        else:
            return point1, point2, point3

    def draw_shape(self, qp):
        if self.anchor_kwargs != None:
            self.update_anchors()
        qp = self.apply_transform(qp)
        if self.show:
            point1, point2, point3 = self._cal_corner_point_coordinates()
            polygon = QPolygonF()
            polygon.append(point1)
            polygon.append(point2)
            polygon.append(point3)
            qp.drawPolygon(polygon)
        self.text_label(qp)

    def text_label(self, qp):
        labels = self.labels
        decoration = self.text_decoration
        point1, point2, point3 = self._cal_corner_point_coordinates(False)
        qp.save()
        qp.translate(*self.rot_center)
        qp.rotate(-self.transformation['rotate'])
        qp.translate(*[-each for each in self.rot_center])
        for i, text in enumerate(labels["text"]):
            anchor = labels["anchor"][i]
            if labels["decoration"] == None:
                decoration = self.text_decoration
            else:
                if type(labels["decoration"]) == list and len(
                    labels["decoration"]
                ) == len(labels["text"]):
                    decoration = labels["decoration"][i]
                else:
                    decoration = self.text_decoration
            alignment = decoration["alignment"]
            padding = decoration["padding"]
            text_color = decoration["text_color"]
            font_size = decoration["font_size"]

            qp.setPen(QColor(*text_color))
            qp.setFont(QFont("Decorative", font_size))
            text_bound_rect = qp.fontMetrics().boundingRect(
                QRect(), Qt.AlignCenter, text
            )
            w_txt, h_txt = text_bound_rect.width(), text_bound_rect.height()
            if anchor == "left":
                x, y = point2
            elif anchor == "right":
                x, y = point3
            elif anchor == "top":
                x, y = point1
            elif anchor == "bottom":
                x, y = (point2 + point3) / 2
            elif anchor == "center":
                x, y = (point2 + point3) / 2
                y = y - abs(point1[1] - y) / 2
            else:
                if anchor in self.anchors:
                    x, y = self.anchors[anchor]
            self._draw_text(
                qp,
                alignment,
                text,
                anchor,
                x,
                y,
                0,
                0,
                w_txt,
                h_txt,
                padding,
                labels["orientation"][i],
            )
            qp.restore()
            qp.save()

    def calculate_shape_boundary(self):
        three_corners = self._cal_corner_point_coordinates(False)
        three_corners = [
            np.array(each) + np.array(self.transformation["translate"])
            for each in three_corners
        ]
        rot_center = np.array(self.rot_center) + np.array(
            self.transformation["translate"]
        )
        three_corners = rotate_multiple_points(
            three_corners, rot_center, self.transformation["rotate"]
        )
        return (
            int(three_corners[:, 0].min()),
            int(three_corners[:, 0].max()),
            int(three_corners[:, 1].min()),
            int(three_corners[:, 1].max()),
        )

    def compute_center_from_dim(self, apply_translate=True):
        x, y, edge, ang = self.dim_pars
        p1, p2, p3 = self._cal_corner_point_coordinates(False)
        r = edge**2 / 2 / abs(p3[1] - p1[1])
        # geometry rot center
        x, y = np.array(p1) + [0, r]
        if apply_translate:
            return (
                x + self.transformation["translate"][0] + self.transformation['translate_offset'][0],
                y + self.transformation["translate"][1] + self.transformation['translate_offset'][1],
            )
        else:
            return x, y

    def make_anchors(self, num_of_anchors_on_each_side=4, include_corner=True):
        # num_of_anchors_on_each_side: exclude corners
        super().make_anchors(
            num_of_anchors_on_each_side=num_of_anchors_on_each_side,
            include_corner=include_corner,
        )
        edge, ang = self.dim_pars[2:]
        ang = math.radians(ang / 2)
        bottom_edge = edge * math.sin(ang) * 2
        height = edge * math.cos(ang)
        if not include_corner:
            w_step, h_step = bottom_edge / (num_of_anchors_on_each_side + 1), height / (
                num_of_anchors_on_each_side + 1
            )
        else:
            assert num_of_anchors_on_each_side > 2, "At least two achors at each edge"
            w_step, h_step = bottom_edge / (num_of_anchors_on_each_side - 1), height / (
                num_of_anchors_on_each_side - 1
            )

        p1, p2, p3 = self._cal_corner_point_coordinates(False)
        anchors = {}
        for i in range(num_of_anchors_on_each_side):
            if not include_corner:
                anchors[f"anchor_left_{i}"] = np.array(p1) + [
                    -(i + 1) * h_step * math.tan(ang),
                    (i + 1) * h_step,
                ]
                anchors[f"anchor_bottom_{i}"] = np.array(p2) + [(i + 1) * w_step, 0]
                anchors[f"anchor_right_{i}"] = np.array(p1) + [
                    (i + 1) * h_step * math.tan(ang),
                    (i + 1) * h_step,
                ]
            else:
                anchors[f"anchor_left_{i}"] = np.array(p1) + [
                    -i * h_step * math.tan(ang),
                    i * h_step,
                ]
                anchors[f"anchor_bottom_{i}"] = np.array(p2) + [i * w_step, 0]
                anchors[f"anchor_right_{i}"] = np.array(p1) + [
                    i * h_step * math.tan(ang),
                    i * h_step,
                ]
        for each in anchors:
            anchors[each] = anchors[each] + np.array(self.transformation["translate"])
        self.anchors = anchors

    def calculate_orientation_length(self, orientation="top", ref_anchor=None):
        if orientation == "cen":
            return 1
        cen = self.compute_center_from_dim(False)
        p1, p2, p3 = self._cal_corner_point_coordinates(False)
        w, h = np.array(self.dim_pars[2:])
        if orientation == "top":
            return abs(cen[1] - p1[1])
        elif orientation == "bottom":
            return abs(cen[1] - p2[1])
        elif orientation in ["left", "right"]:
            return abs(cen[0] - p1[0])
        else:
            if orientation in self.anchors:
                if ref_anchor == None:
                    return np.linalg.norm(
                        np.array(self.anchors[orientation])
                        - np.array(self.compute_center_from_dim(apply_translate=False))
                    )
                else:
                    return 1
            else:
                raise KeyError("No such orientation key!")

    def check_pos(self, x, y):
        p1, p2, p3 = self._cal_corner_point_coordinates(False)
        pos_ = rotate_multiple_points(
            [(x, y)],
            np.array(self.rot_center) + np.array(self.transformation["translate"]) + np.array(self.transformation["translate_offset"]),
            -self.transformation["rotate"],
        )
        pos_ = np.array(pos_) - np.array(self.transformation["translate"]) + np.array(self.transformation["translate_offset"])
        x_, y_ = pos_
        if (p2[0] <= x_ <= p3[0]) and (p1[1] <= y_ <= p2[1]):
            return True
        else:
            return False


class trapezoid(baseShape):
    def __init__(
        self,
        dim=[100, 100, 40, 60, 50],
        rotation_center=None,
        decoration_cursor_off=DECORATION_UPON_CURSOR_OFF,
        decoration_cursor_on=DECORATION_UPON_CURSOR_ON,
        transformation={"rotate": 0, "translate": (0, 0), 'translate_offset':[0,0], "scale": 1},
        text_decoration=DECORATION_TEXT_DEFAULT,
        labels={"text": [], "anchor": [], "orientation": [], "decoration": None},
    ):
        super().__init__(
            dim=dim,
            rotation_center=rotation_center,
            decoration_cursor_off=decoration_cursor_off,
            decoration_cursor_on=decoration_cursor_on,
            transformation=transformation,
            text_decoration=text_decoration,
            lables=labels,
        )

    def scale(self, sf):
        sf_norm = sf / self.transformation["scale"]
        self.dim_pars = (
            np.array(self.dim_pars) * [1, 1, sf_norm, sf_norm, sf_norm]
        ).astype(int)
        self.transformation["scale"] = sf
        if self.anchor_kwargs != None:
            self.update_anchors()

    def _cal_corner_point_coordinates(self, return_type_is_qpointF=True):
        edge_lenth_top = self.dim_pars[-3]
        edge_lenth_bottom = self.dim_pars[-2]
        height = self.dim_pars[-2]
        dx_top = edge_lenth_top / 2
        dx_bottom = edge_lenth_bottom / 2
        dy = height / 2
        point1 = (np.array(self.dim_pars[0:2]) + np.array([-dx_top, -dy])).astype(int)
        point2 = (np.array(self.dim_pars[0:2]) + np.array([dx_top, -dy])).astype(int)
        point3 = (np.array(self.dim_pars[0:2]) + np.array([-dx_bottom, dy])).astype(int)
        point4 = (np.array(self.dim_pars[0:2]) + np.array([dx_bottom, dy])).astype(int)
        if return_type_is_qpointF:
            return (
                QPointF(*point1),
                QPointF(*point2),
                QPointF(*point3),
                QPointF(*point4),
            )
        else:
            return point1, point2, point3, point4

    def draw_shape(self, qp):
        if self.anchor_kwargs != None:
            self.update_anchors()
        qp = self.apply_transform(qp)
        if self.show:
            point1, point2, point3, point4 = self._cal_corner_point_coordinates()
            polygon = QPolygonF()
            polygon.append(point1)
            polygon.append(point2)
            polygon.append(point4)
            polygon.append(point3)
            qp.drawPolygon(polygon)
        self.text_label(qp)

    def _get_width_height(self):
        # (length_top + length_bottom)/2, height
        return max([self.dim_pars[2], self.dim_pars[3]]), self.dim_pars[-1]

    def text_label(self, qp):
        labels = self.labels
        decoration = self.text_decoration
        point1, point2, point3, point4 = self._cal_corner_point_coordinates(False)
        qp.save()
        for i, text in enumerate(labels["text"]):
            anchor = labels["anchor"][i]
            if labels["decoration"] == None:
                decoration = self.text_decoration
            else:
                if type(labels["decoration"]) == list and len(
                    labels["decoration"]
                ) == len(labels["text"]):
                    decoration = labels["decoration"][i]
                else:
                    decoration = self.text_decoration
            alignment = decoration["alignment"]
            padding = decoration["padding"]
            text_color = decoration["text_color"]
            font_size = decoration["font_size"]

            qp.setPen(QColor(*text_color))
            qp.setFont(QFont("Decorative", font_size))
            text_bound_rect = qp.fontMetrics().boundingRect(
                QRect(), Qt.AlignCenter, text
            )
            w_txt, h_txt = text_bound_rect.width(), text_bound_rect.height()
            if anchor == "left":
                x, y = (point1 + point3) / 2
            elif anchor == "right":
                x, y = (point2 + point4) / 2
            elif anchor == "top":
                x, y = self.dim_pars[0:2]
            elif anchor == "bottom":
                x, y = (point3 + point4) / 2
            elif anchor == "center":
                x, y = self.dim_pars[0:2]
            else:
                if anchor in self.anchors:
                    x, y = self.anchors[anchor]
            x, y = self.dim_pars[0:2]
            y = y - self.dim_pars[-1] / 2
            x = x - max(self.dim_pars[2:4]) / 2
            w, h = self._get_width_height()
            self._draw_text(
                qp,
                alignment,
                text,
                anchor,
                x,
                y,
                w,
                h,
                w_txt,
                h_txt,
                padding,
                labels["orientation"][i],
            )
            qp.restore()
            qp.save()

    def calculate_shape_boundary(self):
        four_corners = self._cal_corner_point_coordinates(False)
        four_corners = [
            np.array(each) + np.array(self.transformation["translate"])
            for each in four_corners
        ]
        rot_center = np.array(self.rot_center) + np.array(
            self.transformation["translate"]
        )
        four_corners = rotate_multiple_points(
            four_corners, rot_center, self.transformation["rotate"]
        )
        return (
            int(four_corners[:, 0].min()),
            int(four_corners[:, 0].max()),
            int(four_corners[:, 1].min()),
            int(four_corners[:, 1].max()),
        )

    def compute_center_from_dim(self, apply_translate=True):
        x, y, *_ = self.dim_pars
        if apply_translate:
            return (
                x + self.transformation["translate"][0] + self.transformation['translate_offset'][0],
                y + self.transformation["translate"][1] + self.transformation['translate_offset'][1],
            )
        else:
            return x, y

    def make_anchors(self, num_of_anchors_on_each_side=4, include_corner=True):
        # num_of_anchors_on_each_side: exclude corners
        super().make_anchors(
            num_of_anchors_on_each_side=num_of_anchors_on_each_side,
            include_corner=include_corner,
        )
        bottom_edge = self.dim_pars[-2]
        top_edge = self.dim_pars[-3]
        height = self.dim_pars[-1]
        ang = math.atan(height / ((top_edge - bottom_edge) / 2))
        if not include_corner:
            w_step_bottom, w_step_top, h_step = (
                bottom_edge / (num_of_anchors_on_each_side + 1),
                top_edge / (num_of_anchors_on_each_side + 1),
                height / (num_of_anchors_on_each_side + 1),
            )
        else:
            assert num_of_anchors_on_each_side > 2, "At least two achors at each edge"
            w_step_bottom, w_step_top, h_step = (
                bottom_edge / (num_of_anchors_on_each_side - 1),
                top_edge / (num_of_anchors_on_each_side - 1),
                height / (num_of_anchors_on_each_side - 1),
            )

        p1, p2, p3, p4 = self._cal_corner_point_coordinates(False)
        anchors = {}
        for i in range(num_of_anchors_on_each_side):
            if not include_corner:
                anchors[f"anchor_left_{i}"] = np.array(p1) + [
                    -(i + 1) * h_step * math.tan(ang),
                    (i + 1) * h_step,
                ]
                anchors[f"anchor_bottom_{i}"] = np.array(p3) + [
                    (i + 1) * w_step_bottom,
                    0,
                ]
                anchors[f"anchor_top_{i}"] = np.array(p1) + [(i + 1) * w_step_top, 0]
                anchors[f"anchor_right_{i}"] = np.array(p2) + [
                    (i + 1) * h_step * math.tan(ang),
                    (i + 1) * h_step,
                ]
            else:
                anchors[f"anchor_left_{i}"] = np.array(p1) + [
                    -i * h_step * math.tan(ang),
                    i * h_step,
                ]
                anchors[f"anchor_bottom_{i}"] = np.array(p3) + [i * w_step_bottom, 0]
                anchors[f"anchor_top_{i}"] = np.array(p1) + [i * w_step_top, 0]
                anchors[f"anchor_right_{i}"] = np.array(p2) + [
                    i * h_step * math.tan(ang),
                    i * h_step,
                ]
        for each in anchors:
            anchors[each] = anchors[each]
        self.anchors = anchors

    def calculate_orientation_length(self, orientation="top", ref_anchor=None):
        if orientation == "cen":
            return 1
        w_top, w_bottom, h = np.array(self.dim_pars[2:])
        if orientation in ["top", "bottom"]:
            return h / 2
        elif orientation in ["left", "right"]:
            return (w_top + w_bottom) / 2
        else:
            if orientation in self.anchors:
                if ref_anchor == None:
                    return np.linalg.norm(
                        np.array(self.anchors[orientation])
                        - np.array(self.compute_center_from_dim(apply_translate=False))
                    )
                else:
                    return 1
            else:
                raise KeyError("No such orientation key!")

    def check_pos(self, x, y):
        p1, p2, p3, p4 = self._cal_corner_point_coordinates(False)
        pos_ = rotate_multiple_points(
            [(x, y)],
            np.array(self.rot_center) + np.array(self.transformation["translate"]) + np.array(self.transformation["translate_offset"]),
            -self.transformation["rotate"],
        )
        pos_ = np.array(pos_) - (np.array(self.transformation["translate"] + np.array(self.transformation["translate_offset"])))
        x_, y_ = pos_
        if (p3[0] <= x_ <= p4[0]) and (p1[1] <= y_ <= p3[1]):
            return True
        else:
            return False


class line(baseShape):
    pass


class ellipse(baseShape):
    pass


class pie(baseShape):
    pass
