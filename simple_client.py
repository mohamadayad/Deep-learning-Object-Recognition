import threaded_detector as td

threaded_detector = td.ThreadedDetector()

info = threaded_detector.information_base.get_latest_image_for(0)

print(info)

