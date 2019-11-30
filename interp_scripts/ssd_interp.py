def clip(value):
  return max(min(1.0, value), 0.0)

for frame_results in detections:
  frame_height = frame_results["frame_height"]
  frame_width = frame_results["frame_width"]
  frame_number = frame_results["frame_id"]

  for i in range(frame_results["detections"].shape[2]):
    confidence = frame_results["detections"][0, 0, i, 2]
    if confidence < 0.5:
      continue

    results.add_box(
      xtl=clip(frame_results["detections"][0, 0, i, 3]) * frame_width,
      ytl=clip(frame_results["detections"][0, 0, i, 4]) * frame_height,
      xbr=clip(frame_results["detections"][0, 0, i, 5]) * frame_width,
      ybr=clip(frame_results["detections"][0, 0, i, 6]) * frame_height,
      label=int(frame_results["detections"][0, 0, i, 1]),
      frame_number=frame_number,
      attributes={
        "confidence": "{:.2f}".format(confidence),
      },
    )
