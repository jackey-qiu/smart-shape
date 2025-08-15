from taurus.qt.qtgui.base import TaurusBaseComponent
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import QObject
import copy
from functools import partial
from ..objectBuilder import object_builder
from ..callbacks.callback_container import *
from ...util.util import findMainWindow


class shapeComposite(TaurusBaseComponent, QObject):

    modelKeys = [TaurusBaseComponent.MLIST]
    model_str_list = []
    updateSignal = pyqtSignal()

    def __init__(
        self,
        shapes,
        parent=None,
        anchor_args=None,
        alignment_pattern=None,
        connection_pattern=None,
        ref_shape_index=None,
        model_index_list=[],
        callbacks_upon_model_change=[],
        callbacks_upon_mouseclick=[],
        callbacks_upon_rightmouseclick=[],
        static_labels=[],
        beam_height_offset=0,
        callbacks={},
        models={},
    ):
        super(QObject, shapeComposite).__init__(self)
        TaurusBaseComponent.__init__(self)
        self.model_ix_start = len(self.model_str_list)
        self._shapes = copy.deepcopy(shapes)
        self._model_shape_index_list = model_index_list
        self._callbacks_upon_model_change = callbacks_upon_model_change
        self._callbacks_upon_left_mouseclick = callbacks_upon_mouseclick
        self._callbacks_upon_right_mouseclick = callbacks_upon_rightmouseclick
        self.ref_shape = (
            self.shapes[ref_shape_index] if ref_shape_index != None else self.shapes[0]
        )
        self.anchor_args = anchor_args
        self.alignment = alignment_pattern
        self.connection = connection_pattern
        self.static_labels = static_labels
        self.lines = None
        self.callbacks = callbacks
        self._models = models
        self.beam_height_offset = beam_height_offset
        self.dx_for_fix_exit = 0
        self.parent = findMainWindow()
        self.build_composite()

    def set_dx_for_fix_exit(self, dx):
        self.dx_for_fix_exit = dx

    def set_beam_height_offset(self, offset):
        self.beam_height_offset = offset

    def copy_object_meta(self):
        return (
            {
                "shapes": self._shapes,
                "anchor_args": self.anchor_args,
                "alignment_pattern": self.alignment,
                "connection_pattern": self.connection,
                "ref_shape_index": self._shapes.index(self.ref_shape),
                "model_index_list": self._model_shape_index_list,
                "beam_height_offset": self.beam_height_offset,
            },
            self.callbacks,
            self._models,
        )

    def unpack_callbacks_and_models(self):
        if len(self._models) == 0:
            return
        models = list(self._models.values())
        self.__class__.model_str_list = self.__class__.model_str_list + list(models)
        self.setModel(self.__class__.model_str_list)
        inx_shape = [int(each) for each in self._models.keys()]
        self.model_shape_index_list = inx_shape
        for ix in inx_shape:
            self.shapes[ix].set_clickable(True)
        self.callbacks_upon_model_change = [
            self._make_callback(each, False)
            for each in self.callbacks["callbacks_upon_model_change"].values()
        ]
        self.callbacks_upon_left_mouseclick = [
            self._make_callback(each, True)
            for each in self.callbacks["callbacks_upon_leftmouse_click"].values()
        ]
        self.callbacks_upon_right_mouseclick = [
            self._make_callback(each, True)
            for each in self.callbacks["callbacks_upon_rightmouse_click"].values()
        ]

    def _make_callback(self, callback_info_list, mouseclick_callback=True):
        if callback_info_list == None or callback_info_list == "None":
            return lambda *kwargs: None
        # if there are multiple callbacks linking to one model
        if type(callback_info_list[0]) == list:

            def call_back_chain(parent, shape, model_value):
                cbs = []
                for callback_info in callback_info_list:
                    cb_str = callback_info[0]
                    cb_args = callback_info[1:]
                    cbs.append(
                        partial(
                            eval(cb_str),
                            **{
                                cb_args[i]: cb_args[i + 1]
                                for i in range(0, len(cb_args), 2)
                            },
                        )
                    )
                for cb in cbs:
                    cb(parent, shape, model_value)

            def call_back_chain_mouseclick(parent, shape, model_value):
                cbs = []
                for callback_info in callback_info_list:
                    cb_str = callback_info[0]
                    cb_args = callback_info[1:]
                    cbs.append(
                        partial(
                            eval(cb_str),
                            **{
                                cb_args[i]: cb_args[i + 1]
                                for i in range(0, len(cb_args), 2)
                            },
                        )
                    )
                for cb in cbs:
                    cb(parent, shape, model_value)

            if mouseclick_callback:
                return call_back_chain_mouseclick
            else:
                return call_back_chain
        # if there is only single callback func
        else:
            cb_str = callback_info_list[0]
            cb_args = callback_info_list[1:]
            return partial(
                eval(cb_str),
                **{cb_args[i]: cb_args[i + 1] for i in range(0, len(cb_args), 2)},
            )

    @property
    def shapes(self):
        return self._shapes

    @property
    def model_shape_index_list(self):
        return self._model_shape_index_list

    @model_shape_index_list.setter
    def model_shape_index_list(self, model_shape_index_list):
        shapes_num = len(self.shapes)
        assert (
            type(model_shape_index_list) == list
        ), "please give a list of model shape index"
        for each in model_shape_index_list:
            assert (
                type(each) == int and each < shapes_num
            ), "index must be integer and smaller than the num of total shape in the composite obj"
        self._model_shape_index_list = model_shape_index_list

    @property
    def callbacks_upon_model_change(self):
        return self._callbacks_upon_model_change

    @callbacks_upon_model_change.setter
    def callbacks_upon_model_change(self, cbs):
        assert len(cbs) == len(
            self.model_shape_index_list
        ), "Length of callbacks must equal to that of model shape index"
        self._callbacks_upon_model_change = {
            ix: cb for ix, cb in zip(self.model_shape_index_list, cbs)
        }

    @property
    def callbacks_upon_left_mouseclick(self):
        return self._callbacks_upon_left_mouseclick

    @callbacks_upon_left_mouseclick.setter
    def callbacks_upon_left_mouseclick(self, cbs):
        assert len(cbs) == len(
            self.model_shape_index_list
        ), "Length of callbacks must equal to that of model shape index"
        self._callbacks_upon_left_mouseclick = {
            ix: cb for ix, cb in zip(self.model_shape_index_list, cbs)
        }

    @property
    def callbacks_upon_right_mouseclick(self):
        return self._callbacks_upon_right_mouseclick

    @callbacks_upon_right_mouseclick.setter
    def callbacks_upon_right_mouseclick(self, cbs):
        assert len(cbs) == len(
            self.model_shape_index_list
        ), "Length of callbacks must equal to that of model shape index"
        self._callbacks_upon_right_mouseclick = {
            ix: cb for ix, cb in zip(self.model_shape_index_list, cbs)
        }

    def build_composite(self):
        self.align_shapes()
        self.make_line_connection()
        self.set_static_labels()

    def set_static_labels(self):
        if len(self.static_labels) == 0:
            return
        else:
            assert len(self.shapes) == len(
                self.static_labels
            ), "num of shapes must match the num of static labels"
            for i, each in enumerate(self.static_labels):
                self.shapes[i].labels = {"text": [each]}

    def set_shape_parent(self):
        for each in self.shapes:
            each.set_parent(self)

    def align_shapes(self):
        if self.alignment == None:
            return
        shape_index = self.alignment["shapes"]
        anchors = self.alignment["anchors"]
        gaps = self.alignment["gaps"]
        gaps_absolute = self.alignment.get("gaps_absolute", False)
        ref_anchors = self.alignment["ref_anchors"]
        assert len(shape_index) == len(
            anchors
        ), "Dimension of shape and anchors does not match!"
        for shapes_, anchors_, gap_, ref_anchors_ in zip(
            shape_index, anchors, gaps, ref_anchors
        ):
            ref_shape, target_shape, *_ = [self.shapes[each] for each in shapes_]
            object_builder.buildTools.align_two_shapes(
                ref_shape, target_shape, anchors_, gap_, ref_anchors_, gaps_absolute
            )
        for shape in self.shapes:
            shape.reset_ref_geometry()

    def make_line_connection(self):
        self.lines = []
        if self.connection == None:
            return
        shape_index = self.connection["shapes"]
        anchors = self.connection["anchors"]
        connect_types = self.connection.get("connect_types", [False] * len(anchors))
        assert len(shape_index) == len(
            anchors
        ), "Dimension of shape and anchors does not match!"
        for shapes_, anchors_, connect_ in zip(shape_index, anchors, connect_types):
            shapes = [self.shapes[each] for each in shapes_]
            lines = object_builder.buildTools.make_line_connection_btw_two_anchors(
                shapes, anchors_, direct_connection=connect_
            )
            self.lines.append(lines)

    def translate(self, vec):
        self.ref_shape.translate(vec)
        self.build_composite()

    def rotate(self, ang):
        self.ref_shape.rotate(ang)
        self.build_composite()

    def scale(self, sf):
        for i, shape in enumerate(self.shapes):
            shape.scale(sf)
        self.build_composite()

    def uponLeftMouseClicked(self, shape_index):
        key = (
            TaurusBaseComponent.MLIST,
            self.model_shape_index_list.index(shape_index) + self.model_ix_start,
        )
        self.callbacks_upon_left_mouseclick[shape_index](
            self.parent, self.shapes[shape_index], self.getModelObj(key=key)
        )

    def uponRightMouseClicked(self, shape_index):
        key = (
            TaurusBaseComponent.MLIST,
            self.model_shape_index_list.index(shape_index) + self.model_ix_start,
        )
        return self.callbacks_upon_right_mouseclick[shape_index](
            self.parent, self.shapes[shape_index], self.getModelObj(key=key)
        )

    def handleEvent(self, evt_src, evt_type, evt_value):
        """reimplemented from TaurusBaseComponent"""
        try:
            for i, _ix in enumerate(self.model_shape_index_list):
                key = (TaurusBaseComponent.MLIST, i + self.model_ix_start)
                if (
                    key not in self.modelKeys
                ):  # this could happen when the setModel step is slower than the event polling
                    return
                if evt_src is self.getModelObj(key=key):
                    self._callbacks_upon_model_change[_ix](
                        self.parent, self.shapes[_ix], evt_value
                    )
                    self.updateSignal.emit()
        except Exception as e:
            self.info("Skipping event. Reason: %s", e)
