#!/usr/bin/env python

import numpy as np
import cv2
import sys


class ObjectTracker(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        ret, self.frame = self.cap.read()
        self.scalling_factor = 0.5
        self.frame = cv2.resize(self.frame, None, fx=self.scalling_factor, fy=self.scalling_factor,
                                interpolation=cv2.INTER_AREA)

        cv2.namedWindow('Object tracker')
        cv2.setMouseCallback('Object tracker', self.mouse_event)

        self.selection = None
        self.drag_start = None
        self.tracking_state = 0

    def mouse_event(self, event, x, y, flags, param):
        x, y = np.int16([x, y])

        # wykrywanie ruchu myszka
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.tracking_state = 0

        if self.drag_start:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                h, w = self.frame.shape[:2]
                x0, y0 = self.drag_start
                x0, y0 = np.maximum(0, np.minimum([x0, y0], [x, y]))
                x1, y1 = np.minimum([w, h], np.maximum([x0, y0], [x, y]))
                self.selection = None

                if x1 - x0 > 0 and y1 - y0 > 0:
                    self.selection = (x0, y0, x1, y1)
            else:
                self.drag_start = None
                if self.selection is not None:
                    self.tracking_state = 1

    # metoda do sledzenia obiektu
    def start_tracking(self):
        while True:
            ret, self.frame = self.cap.read()
            self.frame = cv2.resize(self.frame, None, fx=self.scalling_factor, fy=self.scalling_factor,
                                    interpolation=cv2.INTER_AREA)

            vis = self.frame.copy()

            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV).astype("float32")

            # kolor
            lower = np.array((0., 60., 32.))
            upper = np.array((180., 255., 255.))

            mask = cv2.inRange(hsv, lower, upper)

            if self.selection:
                x0, y0, x1, y1 = self.selection
                self.track_window = (x0, y0, x1 - x0, y1 - y0)
                hsv_roi = hsv[y0:y1, x0:x1]
                mask_roi = mask[y0:y1, x0:x1]

                hist = cv2.calcHist([hsv_roi], [0], mask_roi, [16], [0, 180])

                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                self.hist = hist.reshape(-1)

                vis_roi = vis[y0:y1, x0:x1]
                cv2.bitwise_not(vis_roi, vis_roi)
                cv2.medianBlur(hist, 5)
                vis[mask == 0] = 0

            if self.tracking_state == 1:
                self.selection = None
                prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)

                # prob &= mask

                term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
                track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)

                cv2.ellipse(vis, track_box, (0, 255, 0), 2)
            cv2.imshow('Object tracker', vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    ObjectTracker().start_tracking()
