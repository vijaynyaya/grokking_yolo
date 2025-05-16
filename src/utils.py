import torch


def intersection_over_union(boxes_preds, boxes_labels, box_format="tlbr"):
  """Calculate IoU for two bounding boxes."""
  box1 = boxes_preds[..., :4]
  box2 = boxes_labels[..., :4]

  match box_format:
    case "tlbr":
      pass
    case "cwh":
      dw, dh = box1[3:] / 2
      box1 = [box1[0] - dw, box1[1] - dh, box1[2] + dw, box1[3] + dh]
      dw, dh = box2[3:] / 2
      box2 = [box2[0] - dw, box2[1] - dh, box2[2] + dw, box2[3] + dh]
    case _:
      raise ValueError(f"Unsupported box format '{box_format}'. Expected 'tlbr' or 'cwh'")
  
  
  x1 = torch.max(box1[0], box2[0])
  y1 = torch.max(box1[1], box2[1])
  x2 = torch.min(box1[2], box2[2])
  y2 = torch.min(box1[3], box2[3])

  # .clamp(0) is for the case when they do not intersect
  intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

  box1_area = abs((box1[2] - box1[0]) * (box1[3] - box1[1]))
  box2_area = abs((box2[2] - box2[0]) * (box2[3] - box2[1]))

  return intersection / (box1_area + box2_area - intersection + 1e-6)