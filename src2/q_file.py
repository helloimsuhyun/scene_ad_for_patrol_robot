# q_file.py  (2-question per key, backward-compatible)

QUESTIONS = [
    # change_detected (gate + confirm)
    {"id": "change_detected_1", "text": "Is there any physical object change between the left and right? (Yes/No)", "type": "yesno"},
    {"id": "change_detected_2", "text": "Are the left and right physically different (ignore lighting/shadows)? (Yes/No)", "type": "yesno"},

    # door_change (gate + specific)
    {"id": "door_change_1", "text": "Is there any physical object change between the left and right? (Yes/No)", "type": "yesno"},
    {"id": "door_change_2", "text": "Did the open/closed state of a door or gate change on the right? (Yes/No)", "type": "yesno"},

    # new_object (gate + specific)
    {"id": "new_object_1", "text": "Is there any physical object change between the left and right? (Yes/No)", "type": "yesno"},
    {"id": "new_object_2", "text": "Is there a new, removed, or moved object on the right? (Yes/No)", "type": "yesno"},

    # hazard_damage (gate + damage-only)
    {"id": "hazard_damage_1", "text": "Is there any physical object change between the left and right? (Yes/No)", "type": "yesno"},
    {"id": "hazard_damage_2", "text": "Is any object broken, cracked, damaged, or structurally degraded on the right compared to the left? (Yes/No)", "type": "yesno"},

    # description (2개는 optional)
    {"id": "change_description", "text": "What specifically changed on the right? (e.g., 'cable moved', 'box added')", "type": "short_text"},
]
