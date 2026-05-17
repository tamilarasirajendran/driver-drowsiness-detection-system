
"""
This function converts predicted classes into fatigue stages.
Open eyes and no yawning indicate Alert stage, yawning indicates Mild Fatigue,
and closed eyes indicate Severe Fatigue.
"""

#Defines a function named get_fatigue that takes a label as input
def get_fatigue(label):
    """
    Convert 4-class prediction into 3-level fatigue stage.
    This function converts 4 prediction classes into 3 fatigue levels.
    """

    if label in ["Open", "no_yawn"]:  #Checks whether the label is Open or no_yawn
        return 0, "Alert"  #Returns fatigue level 0 and status Alert

    elif label == "yawn":  #Checks whether the label is yawn
        return 1, "Mild Fatigue" #Returns fatigue level 1 and status Mild Fatigue

    elif label == "Closed":  #Checks whether the label is Closed
        return 2, "Severe Fatigue"  #Returns fatigue level 2 and status Severe Fatigue

    else:
        return -1, "Unknown"