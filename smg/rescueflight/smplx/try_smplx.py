import math
import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

import smplx
import smplx.utils
import sys
import torch

from OpenGL.GL import *
from OpenGL.GLU import *
from timeit import default_timer as timer


# See: https://sites.google.com/site/dlampetest/python/calculating-normals-of-a-triangle-mesh-using-numpy
def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens
    return arr


def draw_frame(vertices, faces):
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(-3, 0, 3, 0, 0, 0, 0, 1, 0)

    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    pos = np.array([-1.0, 1.0, 1.0, 0.0])
    glLightfv(GL_LIGHT0, GL_POSITION, pos)

    # glColor3f(0.0, 1.0, 0.0)
    glColor3f(0.75, 0.75, 0.75)
    # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    # glBegin(GL_TRIANGLES)
    # for i in range(len(faces)):
    #     for j in range(3):
    #         glVertex3f(*vertices[faces[i, j]])
    # glEnd()

    # glVertexPointer(3, GL_FLOAT, 0, vertices)
    # glEnableClientState(GL_VERTEX_ARRAY)
    # glDrawElements(GL_TRIANGLES, len(faces), GL_UNSIGNED_INT, faces)
    # glDisableClientState(GL_VERTEX_ARRAY)

    ###
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    # To render without the index list, we create a flattened array where
    # the triangle indices are replaced with the actual vertices.

    # first we create a single column index array
    tri_index = faces.reshape((-1))
    # then we create an indexed view into our vertices and normals
    va = vertices[faces]
    no = norm[faces]

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, va)
    glNormalPointer(GL_FLOAT, 0, no)
    glDrawArrays(GL_TRIANGLES, 0, len(va) * 3)
    ###

    glDisable(GL_LIGHTING)
    glDisable(GL_COLOR_MATERIAL)

    pygame.display.flip()


def main():
    ###
    pygame.init()
    window_size = (640, 480)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)

    ###
    ## Load SMPL model (here we load the male model)
    ## Make sure path is correct
    # noinspection PyTypeChecker
    model: smplx.SMPL = smplx.create("D:/smplx/models", "smpl", gender="male")

    start = timer()

    ## Assign random pose and shape parameters
    # m.pose[:] = np.random.rand(m.pose.size) * .2
    # m.betas[:] = np.random.rand(m.betas.size) * .03
    # m.pose[:] = np.zeros(m.pose.size)
    # m.betas[:] = np.zeros(m.betas.size)

    end = timer()

    print("{}s".format(end - start))

    start = timer()

    # print(m.r)
    # print(len(m.r))
    # print(m.f)

    end = timer()
    print("{}s".format(end - start))

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (window_size[0] / window_size[1]), 0.1, 1000.0)

    angle = 0.0
    direction = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        if direction == 0:
            if angle < 60.0:
                angle += 0.5
            else:
                angle = 60.0
                direction = 1
        else:
            if angle > 0.0:
                angle -= 0.5
            else:
                angle = 0.0
                direction = 0

        start = timer()

        body_pose: np.ndarray = np.zeros(model.NUM_BODY_JOINTS * 3, dtype=np.float32)
        body_pose[3:6] = -np.array([1, 0, 0]) * angle * math.pi / 180
        body_pose[6:9] = np.array([1, 0, 0]) * angle * math.pi / 180
        body_pose[12:15] = np.array([1, 0, 0]) * angle * math.pi / 180
        body_pose[21:24] = np.array([1, 0, 0]) * angle / 2 * math.pi / 180
        body_pose[48:51] = -np.array([1, 0, 0]) * math.pi / 2
        body_pose[54:57] = np.array([0, 0, 1]) * math.pi / 2

        output: smplx.utils.SMPLOutput = model(
            betas=None,
            body_pose=torch.from_numpy(body_pose).unsqueeze(dim=0),
            return_verts=True
        )
        vertices: np.ndarray = output.vertices.detach().cpu().numpy().squeeze()
        draw_frame(vertices, model.faces)

        end = timer()
        print("{}s".format(end - start))


if __name__ == "__main__":
    main()
