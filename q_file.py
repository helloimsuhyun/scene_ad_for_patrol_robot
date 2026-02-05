QUESTION_PAIRS = [
    # 1. Hazard (파손 전용)
    {
        "id": "hazard",
        "questions": [
            "Is there any physical object change between the left and right? (Yes/No)",
            "Is any object broken, cracked, damaged, or structurally degraded on the right compared to the left? (Yes/No)",
        ],
        "class": "hazard",
    },

    # 2. Door
    {
        "id": "door",
        "questions": [
            "Is there any physical object change between the left and right? (Yes/No)",
            "Did the open or closed state of a door or gate change? (Yes/No)",
        ],
        "class": "door",
    },

    # 3. Object change (이동/추가/제거)
    {
        "id": "object_change",
        "questions": [
            "Is there any physical object change between the left and right? (Yes/No)",
            "Is there a new, removed, or moved object on the right? (Yes/No)",
        ],
        "class": "object_change",
    },
]
