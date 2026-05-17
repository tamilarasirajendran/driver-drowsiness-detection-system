
def get_fatigue(label):
    """
    Convert 4-class prediction into 3-level fatigue stage.
    """

    if label in ["Open", "no_yawn"]:
        return 0, "Alert"

    elif label == "yawn":
        return 1, "Mild Fatigue"

    elif label == "Closed":
        return 2, "Severe Fatigue"

    else:
        return -1, "Unknown"