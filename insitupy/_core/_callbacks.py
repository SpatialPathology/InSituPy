

# geometry widget
def _update_keys_based_on_geom_type(widget, xdata):
    # retrieve current value
    current_geom_type = widget.geom_type.value
    current_key = widget.key.value

    # get either regions or annotations object
    geom_data = getattr(xdata, current_geom_type.lower())
    widget.key.choices = sorted(geom_data.metadata.keys(), key=str.casefold)

def _update_classes_on_key_change(widget, xdata):
    # get current values for geom_type and key
    current_geom_type = widget.geom_type.value
    current_key = widget.key.value

    # get either regions or annotations object
    geom_data = getattr(xdata, current_geom_type.lower())

    # update annot_class choices
    widget.annot_class.choices = ["all"] + sorted(geom_data.metadata[current_key]['classes'])

def _set_show_names_based_on_geom_type(widget):
    # retrieve current value
    current_geom_type = widget.geom_type.value

    # set the show_names tick box
    if current_geom_type == "Annotations":
        widget.show_names.value = False

    if current_geom_type == "Regions":
        widget.show_names.value = True