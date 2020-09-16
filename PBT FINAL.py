import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar # <--------
from matplotlib.figure import Figure

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        # set up widgets -------------------------------------------------------
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.gca(projection='3d')
        self.toolbar = NavigationToolbar(self.canvas, self)

        # fwhm widgets
        self.fwhm_xEdit = QLineEdit(self)
        self.fwhm_xEdit.setText("15")
        self.fwhm_yEdit = QLineEdit(self)
        self.fwhm_yEdit.setText("15")

        # field size widget
        self.sizeEdit = QLineEdit(self)
        self.sizeEdit.setText("100")

        # spacing widget
        self.spacingEdit = QLineEdit(self)
        self.spacingEdit.setText("10")

        # shape widget
        self.tiltEdit = QLineEdit(self)
        self.tiltEdit.setText("0")

        # button widget
        self.button = QPushButton("Redraw", self)

        # error widgets
        self.width_errorEdit = QLineEdit(self)
        self.width_errorEdit.setText("0")

        self.position_errorEdit = QLineEdit(self)
        self.position_errorEdit.setText("0")

        self.tilt_errorEdit = QLineEdit(self)
        self.tilt_errorEdit.setText("0")

        # labels to show information
        self.doseLabel1 = QLabel(self)
        self.doseLabel2 = QLabel(self)
        self.doseLabel3 = QLabel(self)
        self.doseLabel4 = QLabel(self)

        # set up layout --------------------------------------------------------
        sublayout = QFormLayout()
        sublayout.addRow("FWHM x (mm)", self.fwhm_xEdit)
        sublayout.addRow("FWHM y (mm)", self.fwhm_yEdit)
        sublayout.addRow("Field size (mm x mm)", self.sizeEdit)
        sublayout.addRow("Spacing (mm)", self.spacingEdit)
        sublayout.addRow("Tilt", self.tiltEdit)
        sublayout.addRow(self.doseLabel1)

        sublayout.addRow("Width error SD", self.width_errorEdit)
        sublayout.addRow("Position error SD", self.position_errorEdit)
        sublayout.addRow("Tilt error SD", self.tilt_errorEdit)
        sublayout.addWidget(self.button)

        sublayout.addRow(self.doseLabel2)
        sublayout.addRow(self.doseLabel3)
        sublayout.addRow(self.doseLabel4)

        mainlayout = QGridLayout()
        mainlayout.addWidget(self.toolbar, 0,0)
        mainlayout.addWidget(self.canvas, 1,0)
        mainlayout.addLayout(sublayout, 1,1)

        self.setLayout(mainlayout)
        self.show()

        # draw/redraw the graph ------------------------------------------------
        self.DrawGraph()
        self.button.clicked.connect(self.button_clicked)

    def DrawGraph(self,fwhm_x=15,fwhm_y=15,stop=100,spacing=10, tilt=0,
                  width_error=0, position_error=0, tilt_error=0):
        self.axes.clear()
        start = 0

        # create meshgrid ------------------------------------------------------
        x = np.linspace(float(start), float(stop), 50)
        y = np.linspace(float(start), float(stop), 50)
        xx,yy = np.meshgrid(x,y)

        # generate spot centers ------------------------------------------------
        mu_x = np.arange(float(start),float(stop)+0.01,float(spacing))
        mu_y = np.arange(float(start),float(stop)+0.01,float(spacing))

        random_spot_centers = []
        for m_x in mu_x:
            for m_y in mu_y:
                random_x = np.random.normal(loc=float(m_x), scale=float(fwhm_x)/100*position_error)
                random_y = np.random.normal(loc=float(m_y), scale=float(fwhm_y)/100*position_error)
                random_spot_centers.append((random_x,random_y))

        # GUI labels -----------------------------------------------------------
        self.doseLabel4.setText(f"Number of spots = {len(random_spot_centers)}")

        # create Gaussian functions at each spot center ------------------------
        Z_coords = []
        for mu in random_spot_centers:
            random_fwhm_x = np.random.normal(loc=float(fwhm_x), scale=float(fwhm_x)/100*width_error)
            random_fwhm_y = np.random.normal(loc=float(fwhm_y), scale=float(fwhm_y)/100*width_error)

            sigma_x = (random_fwhm_x/2)/np.sqrt(2*np.log(2))
            sigma_y = (random_fwhm_y/2)/np.sqrt(2*np.log(2))
            sigma_one = (sigma_x + sigma_y)/2
            sigma_two = sigma_one * 2.5

            newer_fwhm = (fwhm_x + fwhm_y/2)
            random_tilt = np.random.normal(loc=int(tilt), scale=int(newer_fwhm)/100*tilt_error)

            covariance_one = np.array([[ (sigma_one**2), random_tilt], [random_tilt,  (sigma_one**2)]])
            covariance_one_det = np.linalg.det(covariance_one)
            covariance_one_inv = np.linalg.inv(covariance_one)
            A = np.sqrt((2*np.pi)**2 * covariance_one_det)

            covariance_two = np.array([[ (sigma_two**2), random_tilt], [random_tilt,  (sigma_two**2)]])
            covariance_two_det = np.linalg.det(covariance_two)
            covariance_two_inv = np.linalg.inv(covariance_two)
            B = np.sqrt((2*np.pi)**2 * covariance_two_det)

            amplitude = 0.96/A
            ninety = (90*amplitude)/100

            pos = np.empty(xx.shape + (2,)) # pack x and y into single 3d array
            pos[:, :, 0] = xx
            pos[:, :, 1] = yy

            # einsum calculates ("x"-mu)T . covariance^-1 . ("x"-mu)
            einsum_one = np.einsum('...k,kl,...l->...', pos-mu, covariance_one_inv, pos-mu)
            einsum_two = np.einsum('...k,kl,...l->...', pos-mu, covariance_two_inv, pos-mu)

            doubleGauss = (0.96 * np.exp(-einsum_one / 2) / A) + (0.04 * np.exp(-einsum_two / 2) / B)
            Z_coords.append(doubleGauss)

            cset = self.axes.contourf(xx, yy, doubleGauss, [ninety, amplitude], zdir='z', offset=0, cmap=cm.plasma)

        Z = np.array(Z_coords)
        s = np.sum(Z,axis=0) # summation of Gaussians

        # find center 60% x,y coords -------------------------------------------
        twenty = float(stop)/100*20
        eighty = float(stop)/100*80
        mask = np.argwhere((xx >= twenty) & (xx <= eighty) & (yy >= twenty) & (yy <= eighty))

        rows = []
        for i in mask:
            rows.append(i[0])
        columns = []
        for j in mask:
            columns.append(j[1])
        newGaus = s[rows, columns] # new values selected

        # dose calculations ----------------------------------------------------
        dose_max = np.max(s) # the max height might be outside center 60%
        dose_min = np.min(newGaus)
        dose_mean = np.mean(newGaus)
        max_deviation = np.max(abs(newGaus - dose_mean))

        homogeneity = (dose_max-dose_min)/(dose_max+dose_min)*100
        formatted_float = "{:.2f}".format(homogeneity)
        self.doseLabel3.setText(f"Homogeneity = {formatted_float}%")

        # plot 3d visualisation ------------------------------------------------
        self.axes.plot_surface(xx, yy, s, rstride=1, cstride=1,
                               antialiased=True, cmap=cm.plasma_r)
        z_format = mtick.PercentFormatter(xmax=dose_max)
        self.axes.zaxis.set_major_formatter(z_format) # sets z-axis to %
        self.axes.set_zticks(np.linspace(0,dose_max,6))
        self.axes.set_xlabel("(mm)")
        self.axes.set_ylabel("(mm)")
        self.axes.view_init(25,-130)
        self.canvas.draw()


    def button_clicked(self):
        # get new user inputs --------------------------------------------------
        new_fwhm_x = self.fwhm_xEdit.text()
        new_fwhm_y = self.fwhm_yEdit.text()
        new_area = self.sizeEdit.text()
        new_spacing = self.spacingEdit.text()
        new_tilt = self.tiltEdit.text()

        new_width_error = self.width_errorEdit.text()
        new_position_error = self.position_errorEdit.text()
        new_tilt_error = self.tilt_errorEdit.text()

        # redraw graph with new parameters -------------------------------------
        self.DrawGraph(fwhm_x=float(new_fwhm_x), fwhm_y=float(new_fwhm_y),
                       stop=float(new_area), spacing=float(new_spacing),
                       tilt=float(new_tilt), width_error=float(new_width_error),
                       position_error=float(new_position_error),
                       tilt_error=float(new_tilt_error))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    Window = Window()
    Window.setWindowTitle("PBT Dose Visualisation")
    Window.show()
    app.processEvents()
    sys.exit(app.exec_())
