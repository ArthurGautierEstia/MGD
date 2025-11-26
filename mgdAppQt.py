import sys
import json
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QPushButton, QLabel, QSlider, QSpinBox, QFileDialog, QLineEdit,
    QDialog, QSpinBox as ConfigSpinBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
pg.setConfigOptions(antialias=True)

# -------------------------------
# Fonctions utilitaires
# -------------------------------
def parse_value(expr):
    try:
        expr = expr.replace("pi", "np.pi")
        return eval(expr, {"np": np})
    except Exception:
        raise ValueError(f"Expression invalide: {expr}")

def get_cell_value(table, row, col, default=0):
    item = table.item(row, col)
    if item and item.text().strip() != "":
        return parse_value(item.text())
    return default

def dh_modified(alpha, d, theta, r):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st, 0, d],
        [st*ca, ct*ca, -sa, -r*sa],
        [st*sa, ct*sa, ca, r*ca],
        [0, 0, 0, 1]
    ])

def correction_6d(T, tx, ty, tz, rx, ry, rz):
    rx, ry, rz = np.radians([rx, ry, rz])
    Rx = np.array([[1,0,0],
                   [0,np.cos(rx),-np.sin(rx)],
                   [0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],
                   [0,1,0],
                   [-np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],
                   [np.sin(rz),np.cos(rz),0],
                   [0,0,1]])
    R = Rz @ Ry @ Rx # Rotation Fixed angles ZYX
    corr = np.eye(4)
    corr[:3,:3] = R
    corr[:3,3] = [tx, ty, tz]
    return T @ corr

# -------------------------------
# Classe pour configurer les limites des axes
# -------------------------------
class AxisLimitsDialog(QDialog):
    def __init__(self, parent, current_limits):
        super().__init__(parent)
        self.setWindowTitle("Configurer les limites d'axes")
        self.setGeometry(100, 100, 400, 400)
        self.current_limits = current_limits
        
        limits_layout = QVBoxLayout()
        
        # Créer une table pour les limites
        self.table_limits = QTableWidget(6, 2)
        self.table_limits.setHorizontalHeaderLabels(["Min (°)", "Max (°)"])
        self.table_limits.horizontalHeader().setDefaultSectionSize(90)
        limits_layout.addWidget(self.table_limits)
        
        # Initialiser la table avec les valeurs actuelles
        for i in range(6):
            # Ajouter le label q1, q2, etc.
            self.table_limits.setVerticalHeaderLabels([f"q{i+1}" for i in range(6)])
            
            # Min value
            min_item = QTableWidgetItem(str(current_limits[i][0]))
            self.table_limits.setItem(i, 0, min_item)
            
            # Max value
            max_item = QTableWidgetItem(str(current_limits[i][1]))
            self.table_limits.setItem(i, 1, max_item)
        
        # Boutons
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("Valider")
        btn_cancel = QPushButton("Annuler")
        
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        limits_layout.addLayout(btn_layout)
        
        self.setLayout(limits_layout)
    
    def get_limits(self):
        """Retourne les nouvelles limites configurées"""
        limits = []
        for i in range(6):
            min_val = int(self.table_limits.item(i, 0).text())
            max_val = int(self.table_limits.item(i, 1).text())
            limits.append((min_val, max_val))
        return limits

# -------------------------------
# Classe principale PyQt5
# -------------------------------
class MGDApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MGD Robot - PyQtGraph")
        self.resize(1400, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Initialiser les limites des axes (par défaut -180 à 180)
        self.axis_limits = [(-180, 180) for _ in range(6)]

        # Zone tables
        tables_layout = QVBoxLayout()
        tables_layout.addWidget(QLabel("Paramètres DH"))

        self.label_robot_name = QLineEdit()
        self.label_robot_name.setText("")
        self.label_robot_name.setReadOnly(False)  # Permet la modification
        tables_layout.addWidget(self.label_robot_name)

        self.table_dh = QTableWidget(6, 4)
        self.table_dh.setHorizontalHeaderLabels(["alpha (rad)", "d (mm)", "theta (rad)", "r (mm)"])
        self.table_dh.horizontalHeader().setDefaultSectionSize(90)
        self.table_dh.cellChanged.connect(self.visualiser_3d)
        #self.table_dh.setFixedHeight(200)
        tables_layout.addWidget(self.table_dh)

        tables_layout.addWidget(QLabel("Corrections 6D"))
        self.table_corr = QTableWidget(6, 6)
        self.table_corr.setHorizontalHeaderLabels(["Tx(mm)", "Ty(mm)", "Tz(mm)", "Rx(°)", "Ry(°)", "Rz(°)"])
        self.table_corr.horizontalHeader().setDefaultSectionSize(70)
        self.table_corr.cellChanged.connect(self.visualiser_3d)

        #self.table_corr.setFixedHeight(150)
        tables_layout.addWidget(self.table_corr)
        layout.addLayout(tables_layout)

        # Sliders + spinboxes

        slider_layout = QVBoxLayout()
        slider_layout.addWidget(QLabel("Coordonnées articulaires"))
        self.sliders_q = []
        self.spinboxes_q = []

        for i in range(6):
            row_layout = QHBoxLayout()
            label = QLabel(f"q{i+1} (°)")

            slider = QSlider(Qt.Horizontal)
            slider.setRange(self.axis_limits[i][0], self.axis_limits[i][1])
            slider.setValue(0)

            spinbox = QSpinBox()
            spinbox.setRange(self.axis_limits[i][0], self.axis_limits[i][1])
            spinbox.setValue(0)

            slider.valueChanged.connect(spinbox.setValue)
            spinbox.valueChanged.connect(slider.setValue)
            slider.valueChanged.connect(self.visualiser_3d)

            row_layout.addWidget(label)
            row_layout.addWidget(slider)
            row_layout.addWidget(spinbox)

            slider_layout.addLayout(row_layout)

            self.sliders_q.append(slider)
            self.spinboxes_q.append(spinbox)
        
        

        # Boutons
        btn_layout = QVBoxLayout()
        self.btn_calc = QPushButton("Calculer MGD")
        self.btn_visual = QPushButton("Visualiser 3D")
        self.btn_step = QPushButton("Affichage pas à pas")
        self.btn_save = QPushButton("Sauvegarder config")
        self.btn_load = QPushButton("Charger config")
        self.btn_limits = QPushButton("Configurer les limites d'axes")

        slider_layout.addWidget(self.btn_calc)
        slider_layout.addWidget(self.btn_visual)
        slider_layout.addWidget(self.btn_step)
        slider_layout.addWidget(self.btn_save)
        slider_layout.addWidget(self.btn_load)
        slider_layout.addWidget(self.btn_limits)


         # Résultat MGD
        self.result_table = QTableWidget(6, 3)
        self.result_table.setHorizontalHeaderLabels(["Pos TCP","Pos TCP Corrigée", "Deltas"])
        self.result_table.setVerticalHeaderLabels(["X (mm)","Y (mm)","Z (mm)", "A (°)","B (°)","C (°)"])
        self.result_table.horizontalHeader().setDefaultSectionSize(80)
        slider_layout.addWidget(self.result_table)
       
        layout.addLayout(slider_layout)

        # Zone viewer PyQtGraph
        viewer_layout = QVBoxLayout()
        self.viewer = gl.GLViewWidget()
        self.viewer.opts['glOptions'] = 'opaque'
        self.viewer.opts['depth'] = True
        self.viewer.setCameraPosition(distance=1500)
        self.viewer.setMinimumSize(600, 400)
        self.viewer.setBackgroundColor(45, 45, 48, 255)  # Gris clair
        self.ajouter_grille()
        viewer_layout.addWidget(self.viewer)

        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("Précédent")
        self.btn_next = QPushButton("Suivant")
        self.btn_prev.setVisible(False)
        self.btn_next.setVisible(False)
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        viewer_layout.addLayout(nav_layout)
        layout.addLayout(viewer_layout)

       

        # Stocker les transormations
        self.dh_matrices=[np.eye(4)]
        self.corrected_matrices=[np.eye(4)]
        self.dh_pos=np.zeros(3)
        self.dh_ori=np.zeros(3)
        self.corrected_pos=np.zeros(3)
        self.corrected_ori=np.zeros(3)
        self.step_index = None

        # Connexions
        self.btn_calc.clicked.connect(self.calculer_mgd)
        self.btn_visual.clicked.connect(self.visualiser_3d)
        self.btn_step.clicked.connect(self.visualiser_step_by_step)
        self.btn_save.clicked.connect(self.sauvegarder_config)
        self.btn_load.clicked.connect(self.charger_config)
        self.btn_prev.clicked.connect(self.afficher_repere_precedent)
        self.btn_next.clicked.connect(self.afficher_repere_suivant)
        self.btn_limits.clicked.connect(self.configurer_limites_axes)
        


        self.matrices_step = []
        

    def lire_parametres(self):
        params = []
        for i in range(6):
            alpha = get_cell_value(self.table_dh, i, 0)
            d     = get_cell_value(self.table_dh, i, 1)
            theta_offset = get_cell_value(self.table_dh, i, 2)
            r     = get_cell_value(self.table_dh, i, 3)
            q_deg = self.spinboxes_q[i].value()
            q = np.radians(q_deg)
            theta = theta_offset + q
            corr = [get_cell_value(self.table_corr, i, j) for j in range(6)]
            params.append((alpha, d, theta, r, corr))
        return params

    def calculer_mgd(self):
        params = self.lire_parametres()
        self.dh_matrices = [np.eye(4)]
        self.corrected_matrices = [np.eye(4)]
        T_dh = np.eye(4)
        T_corrected = np.eye(4)
        for (alpha, d, theta, r, corr) in params:
            T_dh = T_dh @ dh_modified(alpha, d, theta, r)
            T_corrected = T_corrected @ dh_modified(alpha, d, theta, r)
            T_corrected = correction_6d(T_corrected, *corr)
            self.dh_matrices.append(T_dh)
            self.corrected_matrices.append(T_corrected)
        self.dh_pos = T_dh[:3,3]
        self.dh_ori[0] = np.degrees(np.arctan2(T_dh[2,1], T_dh[2,2]))
        self.dh_ori[1] = np.degrees(np.arctan2(-T_dh[2,0], np.sqrt(T_dh[2,1]**2 + T_dh[2,2]**2)))
        self.dh_ori[2]= np.degrees(np.arctan2(T_dh[1,0], T_dh[0,0]))
        self.corrected_pos = T_corrected[:3,3]
        self.corrected_ori[0] = np.degrees(np.arctan2(T_corrected[2,1], T_corrected[2,2]))
        self.corrected_ori[1] = np.degrees(np.arctan2(-T_corrected[2,0], np.sqrt(T_corrected[2,1]**2 + T_corrected[2,2]**2)))
        self.corrected_ori[2]= np.degrees(np.arctan2(T_corrected[1,0], T_corrected[0,0]))
        
        self.result_table.setItem(0, 0, QTableWidgetItem(f"{self.dh_pos[0]:.2f}"))
        self.result_table.setItem(1, 0, QTableWidgetItem(f"{self.dh_pos[1]:.2f}"))
        self.result_table.setItem(2, 0, QTableWidgetItem(f"{self.dh_pos[2]:.2f}"))
        self.result_table.setItem(3, 0, QTableWidgetItem(f"{self.dh_ori[0]:.4f}"))
        self.result_table.setItem(4, 0, QTableWidgetItem(f"{self.dh_ori[1]:.4f}"))
        self.result_table.setItem(5, 0, QTableWidgetItem(f"{self.dh_ori[2]:.4f}"))

        self.result_table.setItem(0, 1, QTableWidgetItem(f"{self.corrected_pos[0]:.2f}"))
        self.result_table.setItem(1, 1, QTableWidgetItem(f"{self.corrected_pos[1]:.2f}"))
        self.result_table.setItem(2, 1, QTableWidgetItem(f"{self.corrected_pos[2]:.2f}"))
        self.result_table.setItem(3, 1, QTableWidgetItem(f"{self.corrected_ori[0]:.4f}"))
        self.result_table.setItem(4, 1, QTableWidgetItem(f"{self.corrected_ori[1]:.4f}"))
        self.result_table.setItem(5, 1, QTableWidgetItem(f"{self.corrected_ori[2]:.4f}"))

        self.result_table.setItem(0, 2, QTableWidgetItem(f"{self.corrected_pos[0] - self.dh_pos[0]:.2f}"))
        self.result_table.setItem(1, 2, QTableWidgetItem(f"{self.corrected_pos[1] - self.dh_pos[1]:.2f}"))
        self.result_table.setItem(2, 2, QTableWidgetItem(f"{self.corrected_pos[2] - self.dh_pos[2]:.2f}"))
        self.result_table.setItem(3, 2, QTableWidgetItem(f"{self.corrected_ori[0] - self.dh_ori[0]:.4f}"))
        self.result_table.setItem(4, 2, QTableWidgetItem(f"{self.corrected_ori[1] - self.dh_ori[1]:.4f}"))
        self.result_table.setItem(5, 2, QTableWidgetItem(f"{self.corrected_ori[2] - self.dh_ori[2]:.4f}"))


    def visualiser_3d(self):
        self.calculer_mgd()
        self.viewer.clear()
        self.ajouter_grille()
        for T in self.dh_matrices:
            self.afficher_repere(T)
        if self.step_index is not None:
            self.afficher_repere_jaune()

    def visualiser_step_by_step(self):
        if self.btn_prev.isVisible() and self.btn_next.isVisible():
            self.viewer.clear()
            self.btn_prev.setVisible(False)
            self.btn_next.setVisible(False)
            self.step_index = None
            self.visualiser_3d()
        else:
            self.step_index = 0
            self.btn_prev.setVisible(True)
            self.btn_next.setVisible(True)
            self.visualiser_3d()
        

    def afficher_repere_suivant(self):
        if self.step_index < len(self.dh_matrices)-1:
            self.step_index += 1
            self.visualiser_3d()

    def afficher_repere_precedent(self):
        if self.step_index > 0:
            self.step_index -= 1
            self.visualiser_3d()

    
    def afficher_repere(self, T, longueur=50):
        """
        Affiche un repère orienté selon la matrice homogène T.
        T : matrice 4x4 (rotation + translation)
        longueur : taille des axes
        """
        origine = T[:3, 3]  # Position
        R = T[:3, :3]       # Rotation (3x3)

        # Calcul des extrémités des axes en tenant compte de la rotation
        axes = [
            np.array([origine, origine + R[:, 0] * longueur]),  # Axe X
            np.array([origine, origine + R[:, 1] * longueur]),  # Axe Y
            np.array([origine, origine + R[:, 2] * longueur])   # Axe Z
        ]

        couleurs = [(255, 0, 0, 1), (0, 255, 0, 1), (0, 0, 255, 1)]  # X=Rouge, Y=Vert, Z=Bleu

        for i, axis in enumerate(axes):
            plt = gl.GLLinePlotItem(pos=axis, color=couleurs[i], width=2)
            self.viewer.addItem(plt)
        

    
    def afficher_repere_jaune(self,longueur=100):
        T= self.dh_matrices[self.step_index]
        origine = T[:3, 3]  # Position
        R = T[:3, :3]       # Rotation (3x3)

        # Calcul des extrémités des axes en tenant compte de la rotation
        axes = [
            np.array([origine, origine + R[:, 0] * longueur]),  # Axe X
            np.array([origine, origine + R[:, 1] * longueur]),  # Axe Y
            np.array([origine, origine + R[:, 2] * longueur])   # Axe Z
        ]

        couleur = (1, 1, 0, 1)  # Jaune pour tous les axes

        for i, axis in enumerate(axes):
            plt = gl.GLLinePlotItem(pos=axis, color=couleur, width=2)
            self.viewer.addItem(plt)

    def afficher_repere_cylindres(self, T, rayon=5, longueur=50):
        origine = T[:3, 3]
        R = T[:3, :3]
        
        couleurs = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
        directions = [R[:, 0], R[:, 1], R[:, 2]]
        
        for i, (direction, couleur) in enumerate(zip(directions, couleurs)):
            # Créer un cylindre orienté
            mesh = gl.MeshData.cylinder(rows=10, cols=20, radius=[rayon, rayon], length=longueur)
            cylinder = gl.GLMeshItem(meshdata=mesh, color=couleur, smooth=True)
            self.viewer.addItem(cylinder)


    def ajouter_grille(self):
        """Ajoute une grille quadrillée au sol selon les axes X et Y"""
        grid = gl.GLGridItem()
        grid.setSize(x=4000, y=4000, z=0)  # Taille de la grille en mm
        grid.setSpacing(x=200, y=200, z=200)  # Espacement des lignes en mm
        grid.setColor((150, 150, 150, 100))  # Couleur grise semi-transparente
        self.viewer.addItem(grid) 

    def sauvegarder_config(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Sauvegarder configuration", "", "JSON Files (*.json)")
        if file_name:
            data = {
                "dh": [[self.table_dh.item(i,j).text() if self.table_dh.item(i,j) else "" for j in range(4)] for i in range(6)],
                "corr": [[self.table_corr.item(i,j).text() if self.table_corr.item(i,j) else "" for j in range(6)] for i in range(6)],
                "q": [spinbox.value() for spinbox in self.spinboxes_q],
                "name": [self.label_robot_name.text()],
                "axis_limits": self.axis_limits
            }
            with open(file_name, "w") as f:
                json.dump(data, f, indent=4)

    def charger_config(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Charger configuration", "", "JSON Files (*.json)")
        if file_name:
            try:
                with open(file_name, "r") as f:
                    data = json.load(f)
                for i in range(6):
                    for j in range(4):
                        self.table_dh.setItem(i,j,QTableWidgetItem(data["dh"][i][j]))
                    for j in range(6):
                        self.table_corr.setItem(i,j,QTableWidgetItem(data["corr"][i][j]))
                    self.spinboxes_q[i].setValue(data["q"][i])
                
                if "name" in data and len(data["name"]) > 0:
                    self.label_robot_name.setText(data["name"][0])
                else:
                    self.label_robot_name.setText("Configuration chargée")
                
                # Charger et appliquer les limites d'axes
                if "axis_limits" in data:
                    self.axis_limits = data["axis_limits"]
                    self.mettre_a_jour_limites_axes()

            except Exception as e:
                print(f"Erreur lors du chargement: {e}")

    def mettre_a_jour_limites_axes(self):
        """Applique les limites d'axes aux sliders et spinboxes"""
        for i in range(6):
            min_val, max_val = self.axis_limits[i]
            
            # Mettre à jour le slider
            current_value = self.sliders_q[i].value()
            self.sliders_q[i].setRange(min_val, max_val)
            # Clamp la valeur actuelle si elle dépasse les nouvelles limites
            if current_value < min_val:
                self.sliders_q[i].setValue(min_val)
            elif current_value > max_val:
                self.sliders_q[i].setValue(max_val)
            
            # Mettre à jour le spinbox
            current_spinbox_value = self.spinboxes_q[i].value()
            self.spinboxes_q[i].setRange(min_val, max_val)
            if current_spinbox_value < min_val:
                self.spinboxes_q[i].setValue(min_val)
            elif current_spinbox_value > max_val:
                self.spinboxes_q[i].setValue(max_val)

    def configurer_limites_axes(self):
        """Ouvre le dialogue pour configurer les limites des axes"""
        dialog = AxisLimitsDialog(self, self.axis_limits)
        if dialog.exec_() == QDialog.Accepted:
            # Récupérer les nouvelles limites
            self.axis_limits = dialog.get_limits()
            # Appliquer les nouvelles limites
            self.mettre_a_jour_limites_axes()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MGDApp()
    window.show()
    sys.exit(app.exec_())