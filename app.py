import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import time
import cv2
import dash_bootstrap_components as dbc
from datetime import datetime
import numpy as np
import os
from flask import Flask, Response
from flask_sqlalchemy import SQLAlchemy
from src.object_detector.yolov3 import YoloPeopleDetector
from src.object_detector.postprocessor import PostProcessor
from src.visualization.visualizer import CameraViz
from src.data_feed.data_feeder import ViolationsFeed
from dash.dependencies import Input, Output, State
from PIL import Image
from heatmappy import Heatmapper
server = Flask(__name__)
server.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
server.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
db = SQLAlchemy(server)
app = dash.Dash(__name__, server=server,
                external_stylesheets=[dbc.themes.SUPERHERO])


# database
class Violations(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    violations = db.Column(db.Integer, nullable=False)
    nonviolations = db.Column(db.Integer, nullable=False)
    time = db.Column(db.DateTime, default=datetime.now())


class Sevidx(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sev = db.Column(db.Float, nullable=False)
    time = db.Column(db.DateTime, default=datetime.now())


class Vioxy(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    violocx = db.Column(db.Float, nullable=False)
    violocy = db.Column(db.Float, nullable=False)
    time = db.Column(db.DateTime, default=datetime.now())


def stream_test_local_video(path):
    cap = cv2.VideoCapture(path)
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            outs = net.predict(frame)
            pp = PostProcessor()
            indices, boxes, ids, confs, centers = pp.process_preds(frame, outs)
            cameraviz = CameraViz(indices, frame, ids, confs, boxes, centers)
            cameraviz.draw_pred()
            # feed critical dists and non critical into viofeed
            vf.feed_new(
                (list(cameraviz.critical_dists.keys()), cameraviz.alldists, cameraviz.sev_idx))
            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break


@server.route('/video_feed')
def video_feed():
    return Response(stream_test_local_video(path='data/test_videos/cut3.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')


app.layout = html.Div([dbc.Container(
    [
        html.H1("Real Time Social Distancing Monitor",
                style={'text-align': 'center'}),
        dbc.Row(
            [
                dbc.Col(html.Img(
                    src="/video_feed", height=400, width=800), md=4.5),
                dbc.Col(dcc.Graph(id="violations"))
            ],
            align="start"
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="sev-idx")),
                dbc.Col(dcc.Graph(id="heat-map"))
            ],
            align="start",
        )
    ],
    fluid=True,
),
    dcc.Interval(
    id='interval-component',
    interval=5*1000,  # update graph every n*1000 millisecond
    n_intervals=0
)]
)


@app.callback(Output('violations', 'figure'), [Input('interval-component', 'n_intervals')])
def update_violations_graph(n):
    # plotting variables
    t, violist, nonviolist = [], [], []
    vio, nonvio, _, _, _ = vf.get_feed()  # get n frames accumulation
    # insert new record
    db.session.add(Violations(
        violations=vio, nonviolations=nonvio, time=datetime.now()))
    db.session.commit()
    # query all feed to plot
    rows = Violations.query.all()
    for r in rows:
        t.append(r.time)
        violist.append(r.violations)
        nonviolist.append(r.nonviolations)
    fig1 = go.Figure(data=[
        go.Bar(name='Violations', x=t, y=violist),
        go.Bar(name='Non Violations', x=t, y=nonviolist)
    ])
    fig1.update_layout(
        barmode='stack', title_text='Violations VS Non Violations Graph')
    vf.clear_feed()
    return fig1


@app.callback(Output('sev-idx', 'figure'), [Input('interval-component', 'n_intervals')])
def update_sevidx_graph(n):
    # plotting variables
    t, sevidx = [], []
    _, _, sev, _, _ = vf.get_feed()  # get n frames accumulation
    # insert new record
    db.session.add(Sevidx(sev=sev, time=datetime.now()))
    db.session.commit()
    # query all feed to plot
    rows = Sevidx.query.all()
    for r in rows:
        t.append(r.time)
        sevidx.append(r.sev)
    fig2 = go.Figure(data=[
        go.Line(name='Severity Index', x=t, y=sevidx)
    ])
    fig2.update_layout(title_text='Severity Index Over Time')
    vf.clear_feed()
    return fig2


@app.callback(Output('heat-map', 'figure'), [Input('interval-component', 'n_intervals')])
def update_heatmap(n):
    _, _, _, viox, vioy = vf.get_feed()
    plotlist = []
    for x, y in zip(viox, vioy):
        db.session.add(Vioxy(violocx=x, violocy=y, time=datetime.now()))
    db.session.commit()
    rows = Vioxy.query.all()
    for r in rows:
        plotlist.append((r.violocx, r.violocy))
    heatmapper = Heatmapper(
        point_strength=0.5, point_diameter=300, opacity=0.35)
    heatmapres = heatmapper.heatmap_on_img(plotlist, heatmap)
    heatmapres.save(heatmap_res)
    vf.clear_feed()
    img_width = 800
    img_height = 400
    scale_factor = 0.5

    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    # fig4.add_trace(
    #     go.Scatter(
    #         x=[0, img_width * scale_factor],
    #         y=[0, img_height * scale_factor],
    #         mode="markers",
    #         marker_opacity=0
    #     )
    # )

    # # Configure axes
    # fig4.update_xaxes(
    #     visible=False,
    #     range=[0, img_width * scale_factor]
    # )

    # fig4.update_yaxes(
    #     visible=False,
    #     range=[0, img_height * scale_factor],
    #     # the scaleanchor attribute ensures that the aspect ratio stays constant
    #     scaleanchor="x"
    # )

    # Add image
    fig4.add_layout_image(
        dict(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=heatmap_res)
    )

    # Configure other layout
    fig4.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )
    return fig4


if __name__ == '__main__':
    vf = ViolationsFeed()
    # init vio graph
    fig1 = go.Figure()
    fig2 = go.Figure()
    fig3 = go.Figure()
    fig4 = go.Figure()
    # heatmap Static Image
    static_heatmap = 'data/test_images/heat.png'
    heatmap_res = 'data/test_images/heatres.png'
    heatmap = Image.open(static_heatmap)
    # init and load yolo network
    net = YoloPeopleDetector()
    net.load_network()
    app.run_server(debug=True)
