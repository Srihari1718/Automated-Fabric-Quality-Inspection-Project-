def expert_validation(confidence, threshold=80):
    if confidence < threshold:
        return "Expert Review Required"
    return "Approved by System"
