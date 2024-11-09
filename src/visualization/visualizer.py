import cv2
import numpy as np
from matplotlib import pyplot as plt
from io import BytesIO
from PIL import Image

class Visualizer:
    def __init__(self, critical_line_color=(0, 0, 255), critical_line_thickness=5):
        self.critical_line_color = critical_line_color
        self.critical_line_thickness = critical_line_thickness

    def draw_pred(self):
        pass

    def plot_to_image(self, fig):
        """Convert a Matplotlib figure to an image for OpenCV display"""
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img = np.array(Image.open(buf))
        buf.close()
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

class CameraViz(Visualizer):
    def __init__(self, nmsboxes, frame, classIds, confs, boxes, centers, labelpath='yolo_weights/coco.names',
                 detected_object_rect_color=(255, 178, 50), detected_object_rect_thickness=3,
                 label_font=cv2.FONT_HERSHEY_SIMPLEX, label_fontscale=0.5, label_font_thickness=1,
                 label_rect_color=(255, 255, 255), label_text_color=(0, 0, 0), meter_fontscale=1, meter_font_thickness=2,
                 meter_text_color=(255, 0, 0)):
        super().__init__()
        self._labelpath = labelpath
        self._labels = open(self._labelpath).read().strip().split("\n")
        self.detected_object_rect_color = detected_object_rect_color
        self.detected_object_rect_thickness = detected_object_rect_thickness
        self.label_font = label_font
        self.label_fontscale = label_fontscale
        self.label_font_thickness = label_font_thickness
        self.label_rect_color = label_rect_color
        self.label_text_color = label_text_color
        self.meter_fontscale = meter_fontscale
        self.meter_font_thickness = meter_font_thickness
        self.meter_text_color = meter_text_color
        self.__nmsboxes = nmsboxes
        self.__frame = frame
        self.__boxes = boxes
        self.__classIds = classIds
        self.__confs = confs
        self.__centers = centers
        self.critical_dists = {}
        self.alldists = []
        self.sev_idx = 0.0

    def draw_dashboard(self, frame, viofeed, nonviofeed, sevidx, violocationsx, violocationsy):
        """
        Draw dashboard with graphs on the frame, updating dynamically.
        """
        # Dashboard layout
        dashboard_height = 300
        total_width = frame.shape[1]

        # Plot: Violations Histogram (Real-time update)
        fig, ax = plt.subplots(figsize=(4, 2), dpi=50)
        ax.hist(viofeed, bins=5, color='red', alpha=0.7, label='Violations')
        ax.set_title("Violations Histogram")
        ax.set_xlabel("Violation Instances")
        ax.set_ylabel("Frequency")
        hist_img = self.plot_to_image(fig)
        plt.close(fig)

        # Resize hist_img to fit within the dashboard height and frame width
        hist_img = cv2.resize(hist_img, (frame.shape[1] // 4, dashboard_height))  # Resize width to fit 1/4th of frame width

        # Plot: Non-Violations Histogram (Real-time update)
        fig, ax = plt.subplots(figsize=(4, 2), dpi=50)
        ax.hist(nonviofeed, bins=5, color='green', alpha=0.7, label='Non-Violations')
        ax.set_title("Non-Violations Histogram")
        ax.set_xlabel("Non-Violation Instances")
        ax.set_ylabel("Frequency")
        nonvio_img = self.plot_to_image(fig)
        plt.close(fig)

        # Resize nonvio_img
        nonvio_img = cv2.resize(nonvio_img, (frame.shape[1] // 4, dashboard_height))  # Resize width to fit 1/4th of frame width

        # Plot: Severity Index Trend (Real-time update)
        fig, ax = plt.subplots(figsize=(4, 2), dpi=50)
        ax.plot(sevidx, color='blue', label='Severity Index')
        ax.set_title("Severity Index Over Time")
        ax.set_xlabel("Time Intervals")
        ax.set_ylabel("Severity Index")
        sev_img = self.plot_to_image(fig)
        plt.close(fig)

        # Resize sev_img
        sev_img = cv2.resize(sev_img, (frame.shape[1] // 4, dashboard_height))  # Resize width to fit 1/4th of frame width

        # Plot: Violations Location Scatter (Real-time update)
        fig, ax = plt.subplots(figsize=(4, 2), dpi=50)
        ax.scatter(violocationsx, violocationsy, c='orange', label='Violation Locations')
        ax.set_title("Violation Locations")
        ax.set_xlabel("X Coordinates")
        ax.set_ylabel("Y Coordinates")
        loc_img = self.plot_to_image(fig)
        plt.close(fig)

        # Resize loc_img
        loc_img = cv2.resize(loc_img, (frame.shape[1] // 4, dashboard_height))  # Resize width to fit 1/4th of frame width

        # Ensure the total width is enough for all images
        if total_width < (hist_img.shape[1] + nonvio_img.shape[1] + sev_img.shape[1] + loc_img.shape[1]):
            print("Frame width is too small to fit the dashboard layout. Adjusting layout.")
            return frame  # Skip drawing dashboard if width is insufficient

        # Overlay graphs onto the frame (drawing all 4 images side by side)
        frame[frame.shape[0] - dashboard_height:frame.shape[0], :hist_img.shape[1]] = hist_img
        frame[frame.shape[0] - dashboard_height:frame.shape[0], hist_img.shape[1]:hist_img.shape[1] + nonvio_img.shape[1]] = nonvio_img
        frame[frame.shape[0] - dashboard_height:frame.shape[0], hist_img.shape[1] + nonvio_img.shape[1]:hist_img.shape[1] + nonvio_img.shape[1] + sev_img.shape[1]] = sev_img
        frame[frame.shape[0] - dashboard_height:frame.shape[0], hist_img.shape[1] + nonvio_img.shape[1] + sev_img.shape[1]:] = loc_img

        return frame


    def draw_pred(self, frame, viofeed, nonviofeed, sevidx, violocationsx, violocationsy):
        # Draw predictions on the frame
        for i in self.__nmsboxes:
            box = self.__boxes[i]
            left, top, width, height = box[0], box[1], box[2], box[3]
            cv2.rectangle(frame, (left, top), (left + width, top + height), self.detected_object_rect_color, self.detected_object_rect_thickness)
            label = f"{self._labels[self.__classIds[i]]}: {self.__confs[i]:.2f}"
            cv2.putText(frame, label, (left, top - 10), self.label_font, self.label_fontscale, self.label_text_color, self.label_font_thickness)

        # Add dashboard
        frame = self.draw_dashboard(frame, viofeed, nonviofeed, sevidx, violocationsx, violocationsy)
        return frame
