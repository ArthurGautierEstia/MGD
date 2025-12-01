import sys
import json
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QTableWidget, QTableWidgetItem, QPushButton, QLabel, QSlider, QSpinBox, QFileDialog, QLineEdit, QCheckBox, QMessageBox,
    QDialog, QSpinBox as ConfigSpinBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
from stl import mesh
from scipy.spatial.transform import Rotation as R

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
# Classe pour le paramétrage des axes
# -------------------------------
class AxisLimitsDialog(QDialog):
    def __init__(self, parent, current_limits, home_position=None):
        super().__init__(parent)
        self.setWindowTitle("Paramètrage des axes")
        self.setGeometry(100, 100, 500, 400)
        self.current_limits = current_limits
        self.home_position = home_position if home_position else [0, -90, 90, 0, 90, 0]
        
        limits_layout = QVBoxLayout()
        
        # Créer une table pour les limites avec colonne Home Position
        self.table_limits = QTableWidget(6, 3)
        self.table_limits.setHorizontalHeaderLabels(["Min (°)", "Max (°)", "Home (°)"])
        self.table_limits.horizontalHeader().setDefaultSectionSize(100)
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
            
            # Home Position
            home_item = QTableWidgetItem(str(self.home_position[i]))
            self.table_limits.setItem(i, 2, home_item)
        
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
        
        # Centrer la fenêtre sur l'écran
        self.center_on_screen()
    
    def center_on_screen(self):
        """Centre la fenêtre de dialogue sur l'écran"""
        screen_geometry = QApplication.primaryScreen().geometry()
        dialog_width = self.width()
        dialog_height = self.height()
        x = (screen_geometry.width() - dialog_width) // 2
        y = (screen_geometry.height() - dialog_height) // 2
        self.move(x, y)
    
    def get_limits(self):
        """Retourne les nouvelles limites configurées"""
        limits = []
        for i in range(6):
            min_val = int(self.table_limits.item(i, 0).text())
            max_val = int(self.table_limits.item(i, 1).text())
            limits.append((min_val, max_val))
        return limits
    
    def get_home_position(self):
        """Retourne la position home configurée"""
        home_pos = []
        for i in range(6):
            home_val = int(self.table_limits.item(i, 2).text())
            home_pos.append(home_val)
        return home_pos

# -------------------------------
# Classe principale PyQt5
# -------------------------------
class MGDApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MGD Robot Compensator")
        self.resize(2000, 1100)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Initialiser les limites des axes (par défaut -180 à 180)
        self.axis_limits = [(-180, 180) for _ in range(6)]
        
        # Initialiser la position home (par défaut (0, -90, 90, 0, 90, 0))
        self.home_position = [0, -90, 90, 0, 90, 0]

        # Zone tables
        tables_layout = QVBoxLayout()

        th_layout = QGridLayout()
        tables_layout.addWidget(QLabel("Configuration robot (DHM nominaux)"))
        
        self.label_robot_name_th = QLineEdit()
        self.label_robot_name_th.setReadOnly(False)  # Permet la modification
        th_layout.addWidget(self.label_robot_name_th, 0, 0)

        self.cad_cb = QCheckBox("CAD")
        self.cad_cb.stateChanged.connect(self.basculer_visibilite_robot)
        th_layout.addWidget(self.cad_cb, 0, 1)

        self.btn_load_th = QPushButton("Charger")
        th_layout.addWidget(self.btn_load_th, 0, 2)

        self.btn_save_th = QPushButton("Sauvegarder")
        th_layout.addWidget(self.btn_save_th, 0, 3)
        
        tables_layout.addLayout(th_layout)

        self.table_dh = QTableWidget(6, 4)
        self.table_dh.setHorizontalHeaderLabels(["alpha (°)", "d (mm)", "theta (°)", "r (mm)"])
        self.table_dh.horizontalHeader().setDefaultSectionSize(90)
        self.table_dh.cellChanged.connect(self.visualiser_3d)
        tables_layout.addWidget(self.table_dh)

        me_layout = QGridLayout()
        tables_layout.addWidget(QLabel("Mesures robot (DHM mesurés)"))

        self.label_robot_name_me = QLineEdit()
        self.label_robot_name_me.setReadOnly(False)  # Permet la modification
        me_layout.addWidget(self.label_robot_name_me, 1, 0)

        self.btn_import_me = QPushButton("Importer")
        me_layout.addWidget(self.btn_import_me, 1, 1)

        self.btn_clear_me = QPushButton("Vider")
        me_layout.addWidget(self.btn_clear_me, 1, 2)

        tables_layout.addLayout(me_layout)

        self.table_me = QTableWidget(6, 4)
        self.table_me.setHorizontalHeaderLabels(["alpha (°)", "d (mm)", "theta (°)", "r (mm)"])
        self.table_me.horizontalHeader().setDefaultSectionSize(90)
        #self.table_me.cellChanged.connect(self.visualiser_3d)
        tables_layout.addWidget(self.table_me)

        self.btn_calculate_corr = QPushButton("Calculer les corrections")
        tables_layout.addWidget(self.btn_calculate_corr)

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
        btn_grid = QGridLayout()

        self.btn_limits = QPushButton("Paramètrage des axes")
        self.btn_home_position = QPushButton("Position home")


        btn_grid.addWidget(self.btn_limits, 0, 0)
        btn_grid.addWidget(self.btn_home_position, 0, 1)
        btn_layout.addLayout(btn_grid)
        slider_layout.addLayout(btn_layout)
        
        self.btn_step = QPushButton("Affichage pas à pas")
        slider_layout.addWidget(self.btn_step)

         # Résultat MGD
        slider_layout.addWidget(QLabel("Positions cartésiennes"))
        self.result_table = QTableWidget(6, 4)
        self.result_table.setHorizontalHeaderLabels(["TCP","TCP Corr", "Ecarts", "Jog"])
        self.result_table.setVerticalHeaderLabels(["X (mm)","Y (mm)","Z (mm)", "A (°)","B (°)","C (°)"])
        self.result_table.horizontalHeader().setDefaultSectionSize(110)

        # Ajouter les boutons + et - dans la colonne 3
        for row in range(6):
            # Créer les boutons
            btn_plus = QPushButton("+")
            btn_minus = QPushButton("-")

            # Connecter les boutons à la fonction d'incrément
            btn_plus.clicked.connect(lambda _, r=row: self.increment_value(r, +1))
            btn_minus.clicked.connect(lambda _, r=row: self.increment_value(r, -1))

            # Créer un layout horizontal pour les deux boutons
            btn_layout = QHBoxLayout()
            btn_layout.addWidget(btn_minus)
            btn_layout.addWidget(btn_plus)
            btn_layout.setContentsMargins(0, 0, 0, 0)

            # Créer un widget conteneur pour le layout
            cell_widget = QWidget()
            cell_widget.setLayout(btn_layout)

            # Insérer le widget dans la cellule (colonne 3 par exemple)
            self.result_table.setCellWidget(row, 3, cell_widget)

        slider_layout.addWidget(self.result_table)

        slider_layout.addWidget(QLabel("Corrections 6D"))
        self.table_corr = QTableWidget(6, 6)
        self.table_corr.setHorizontalHeaderLabels(["Tx(mm)", "Ty(mm)", "Tz(mm)", "Rx(°)", "Ry(°)", "Rz(°)"])
        self.table_corr.horizontalHeader().setDefaultSectionSize(80)
        self.table_corr.cellChanged.connect(self.visualiser_3d)
        slider_layout.addWidget(self.table_corr)

        layout.addLayout(slider_layout)

        # Zone viewer PyQtGraph
        viewer_layout = QVBoxLayout()
        self.viewer = gl.GLViewWidget()
        self.viewer.opts['glOptions'] = 'translucent'
        self.viewer.opts['depth'] = True
        self.viewer.setCameraPosition(distance=2000, elevation=40, azimuth=45)
        self.viewer.setMinimumSize(900, 400)
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

        #Stocker les mesh importés
        self.robot_links = []
        
        # Tracker la configuration actuellement chargée
        self.current_config_file = None

        # Connexions
        self.btn_step.clicked.connect(self.visualiser_step_by_step)

        self.btn_save_th.clicked.connect(self.sauvegarder_config)
        self.btn_load_th.clicked.connect(self.charger_config)
        self.btn_import_me.clicked.connect(self.importer_mesures)
        self.btn_clear_me.clicked.connect(self.vider_mesures)
        self.btn_calculate_corr.clicked.connect(self.calculer_ecarts_dh_me)
        
        self.btn_prev.clicked.connect(self.afficher_repere_precedent)
        self.btn_next.clicked.connect(self.afficher_repere_suivant)
        self.btn_limits.clicked.connect(self.configurer_limites_axes)
        self.btn_home_position.clicked.connect(self.appliquer_home_position)

        self.matrices_step = []
        
    def lire_parametres(self):
        params = []
        for i in range(self.table_dh.rowCount()):
            alpha = np.radians(get_cell_value(self.table_dh, i, 0))
            d     = get_cell_value(self.table_dh, i, 1)
            theta_offset = np.radians(get_cell_value(self.table_dh, i, 2))
            r     = get_cell_value(self.table_dh, i, 3)
            q_deg = self.spinboxes_q[i].value()
            q = np.radians(q_deg)
            theta = theta_offset + q
            corr = [get_cell_value(self.table_corr, i, j) for j in range(6)]
            params.append((alpha, d, theta, r, corr))
        return params

    def lire_parametres_me(self):
        """Lit les paramètres mesurés de la table_me"""
        params = []
        for i in range(self.table_me.rowCount()):
            alpha = np.radians(get_cell_value(self.table_me, i, 0))
            d     = get_cell_value(self.table_me, i, 1)
            theta_offset = np.radians(get_cell_value(self.table_me, i, 2))
            r     = get_cell_value(self.table_me, i, 3)
            q_deg = self.spinboxes_q[i].value()
            q = np.radians(q_deg)
            theta = theta_offset + q
            params.append((alpha, d, theta, r))
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
        self.update_segment_pose()
        
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

    def afficher_repere(self, T, longueur=100):
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
            plt = gl.GLLinePlotItem(pos=axis, color=couleurs[i], width=3, antialias=True)
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
            plt = gl.GLLinePlotItem(pos=axis, color=couleur, width=3, antialias=True)
            self.viewer.addItem(plt)

    def ajouter_grille(self):
        """Ajoute une grille quadrillée au sol selon les axes X et Y"""
        grid = gl.GLGridItem()
        grid.setSize(x=4000, y=4000, z=0)  # Taille de la grille en mm
        grid.setSpacing(x=200, y=200, z=200)  # Espacement des lignes en mm
        grid.setColor((150, 150, 150, 100))  # Couleur grise semi-transparente
        self.viewer.addItem(grid)

    def update_segment_pose(self):
        for i in range(len(self.robot_links)):
            mesh_item = self.robot_links[i]
            T= self.corrected_matrices[i]
            if mesh_item:
                mesh_item.resetTransform()  # Réinitialise la transformation
                R = T[:3, :3]
                pos = T[:3, 3]
                # Appliquer rotations ZYX
                rx = np.degrees(np.arctan2(R[2,1], R[2,2]))
                ry = np.degrees(np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2)))
                rz = np.degrees(np.arctan2(R[1,0], R[0,0]))
                mesh_item.rotate(rx, 1, 0, 0)
                mesh_item.rotate(ry, 0, 1, 0)
                mesh_item.rotate(rz, 0, 0, 1)
                mesh_item.translate(pos[0], pos[1], pos[2])
            self.viewer.addItem(mesh_item)
 
    def sauvegarder_config(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Sauvegarder configuration", "", "JSON Files (*.json)")
        if file_name:
            data = {
                "dh": [[self.table_dh.item(i,j).text() if self.table_dh.item(i,j) else "" for j in range(4)] for i in range(6)],
                "corr": [[self.table_corr.item(i,j).text() if self.table_corr.item(i,j) else "" for j in range(6)] for i in range(6)],
                "q": [spinbox.value() for spinbox in self.spinboxes_q],
                "name": [self.label_robot_name_th.text()],
                "axis_limits": self.axis_limits,
                "home_position": self.home_position
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
                        self.label_robot_name_th.setText(data["name"][0])
                    else:
                        self.label_robot_name_th.setText("Configuration chargée")
                    
                    # Charger et appliquer les limites d'axes
                    if "axis_limits" in data:
                        self.axis_limits = data["axis_limits"]
                        self.mettre_a_jour_limites_axes()
                    
                    # Charger et appliquer la home position
                    if "home_position" in data:
                        self.home_position = data["home_position"]
                    else:
                        self.home_position = [0, -90, 90, 0, 90, 0]
                    
                    # Vérifier si c'est une nouvelle configuration
                    if self.current_config_file != file_name:
                        # Nouvelle configuration : nettoyer et importer les segments
                        self.nettoyer_robot()
                        self.importer_segment()
                        self.current_config_file = file_name
                    # Sinon, c'est la même configuration : pas besoin d'importer les segments à nouveau

                except Exception as e:
                    print(f"Erreur lors du chargement: {e}")

    def nettoyer_robot(self):
        """Supprime tous les mesh items du robot du viewer"""
        for mesh_item in self.robot_links:
            self.viewer.removeItem(mesh_item)
        self.robot_links.clear()

    def basculer_visibilite_robot(self):
        """Bascule la visibilité du robot"""
        if self.cad_cb.isChecked():
            for mesh_item in self.robot_links:
                mesh_item.show()
        else:
            for mesh_item in self.robot_links:
                mesh_item.hide()

    def importer_segment(self):
            """
            Importe un segment du robot à partir d'un fichier STL et l'affiche
            selon la transformation homogène T (4x4).
            """
            kuka_orange = (1.0, 0.4, 0.0, 0.5)  # Couleur orange KUKA
            kuka_black = (0.1, 0.1, 0.1, 0.5)   # Couleur noire KUKA
            kuka_grey = (0.5, 0.5, 0.5, 0.5)    # Couleur grise KUKA
            for i in range(0,7):
                chemin_stl = f"./robot/rocky{i}.stl"
                T=self.corrected_matrices[i]

                if i == 0:
                    kuka_color = kuka_black
                elif i == 6:
                    kuka_color = kuka_grey
                else:
                    kuka_color = kuka_orange
                
                try:
                    # Charger le STL avec numpy-stl
                    stl_mesh = mesh.Mesh.from_file(chemin_stl)
                    
                    # Extraire les sommets et faces
                    verts = stl_mesh.vectors.reshape(-1, 3)
                    faces = np.arange(len(verts)).reshape(-1, 3)
                    
                    # Créer MeshData
                    mesh_data = gl.MeshData(vertexes=verts, faces=faces)

                    # Créer l'objet 3D
                    mesh_item = gl.GLMeshItem(meshdata=mesh_data, smooth=True, color=kuka_color, shader='shaded')

                    # Appliquer la transformation homogène T
                    pos = T[:3, 3]
                    
                    # Rotation : utiliser la matrice T[:3,:3]
                    # GLMeshItem n'accepte pas directement une matrice, donc on applique des rotations successives
                    R = T[:3, :3]
                    rx = np.degrees(np.arctan2(R[2,1], R[2,2]))
                    ry = np.degrees(np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2)))
                    rz = np.degrees(np.arctan2(R[1,0], R[0,0]))
                    mesh_item.rotate(rx, 1, 0, 0)
                    mesh_item.rotate(ry, 0, 1, 0)
                    mesh_item.rotate(rz, 0, 0, 1)
                    mesh_item.translate(pos[0], pos[1], pos[2])
                    
                    # Ajouter au viewer
                    self.robot_links.append(mesh_item)
                    self.viewer.addItem(mesh_item)
                    self.basculer_visibilite_robot()          
                    
                except Exception as e:
                    print(f"Erreur lors de l'import du segment: {e}")

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
        dialog = AxisLimitsDialog(self, self.axis_limits, self.home_position)
        if dialog.exec_() == QDialog.Accepted:
            # Récupérer les nouvelles limites
            self.axis_limits = dialog.get_limits()
            # Récupérer la nouvelle home position
            self.home_position = dialog.get_home_position()
            # Appliquer les nouvelles limites
            self.mettre_a_jour_limites_axes()

    def appliquer_home_position(self):
        """Applique la home position aux sliders et spinboxes"""
        for i in range(6):
            self.sliders_q[i].setValue(self.home_position[i])
            self.spinboxes_q[i].setValue(self.home_position[i])

    def importer_mesures(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Importer mesures", "", "JSON Files (*.json)")
        if file_name:
            try:
                with open(file_name, "r") as f:
                    data = json.load(f)
                for i in range(6):
                    for j in range(4):
                        self.table_me.setItem(i,j,QTableWidgetItem(data["dh"][i][j]))
                if "name" in data and len(data["name"]) > 0:
                    self.label_robot_name_me.setText(data["name"][0])
                else:
                    self.label_robot_name_me.setText("Mesures chargées")

            except Exception as e:
                print(f"Erreur lors de l'importation des mesures: {e}")
        
    def vider_mesures(self):
        self.table_me.clearContents()
        self.label_robot_name_me.setText("")  

    def calculer_ecarts_dh_me(self):
        """Calcule les écarts entre les matrices DH et ME et les affiche dans table_corr"""
        try:
            # Vérifier que table_me contient des données
            if not self.table_me.item(0, 0) or self.table_me.item(0, 0).text() == "":
                return
            
            # Lire les paramètres mesurés
            params_me = self.lire_parametres_me()
            
            T_me = np.eye(4)

            for i, (alpha, d, theta, r) in enumerate(params_me):
                T_dh = self.dh_matrices[i+1]  # Récupérer la matrice DH déjà calculée
                T_me = T_me @ dh_modified(alpha, d, theta, r)
                
                # Écart de position
                delta_pos = T_me[:3, 3] - T_dh[:3, 3]
                # Écart d'orientation : matrice de rotation d'erreur
                R_dh = T_dh[:3, :3]
                R_me = T_me[:3, :3]
                
                # Matrice de rotation d'erreur relative : R_error = R_me @ R_dh^T
                R_error = R_me @ R_dh.T
                # Extraire les angles Euler ZYX de la matrice d'erreur
                angles = R.from_matrix(R_error).as_euler('zyx', degrees=True)
                rx_error, ry_error, rz_error = angles[::-1]  # car ZYX

                # Afficher les écarts dans la table_corr
                # Colonnes: Tx, Ty, Tz, Rx, Ry, Rz
                self.table_corr.setItem(i, 0, QTableWidgetItem(f"{delta_pos[0]:.2f}"))
                self.table_corr.setItem(i, 1, QTableWidgetItem(f"{delta_pos[1]:.2f}"))
                self.table_corr.setItem(i, 2, QTableWidgetItem(f"{delta_pos[2]:.2f}"))
                self.table_corr.setItem(i, 3, QTableWidgetItem(f"{rx_error:.4f}"))
                self.table_corr.setItem(i, 4, QTableWidgetItem(f"{ry_error:.4f}"))
                self.table_corr.setItem(i, 5, QTableWidgetItem(f"{rz_error:.4f}"))
        
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors du calcul des écarts: {e}")

    def increment_value(self, row, delta):
        """Incrémente ou décrémente la valeur dans la colonne 0 de la ligne donnée"""
        item_tcp = self.result_table.item(row, 0)
        item_tcp_corr = self.result_table.item(row, 1)

        if item_tcp is None:
            item_tcp = QTableWidgetItem("0.00")
            self.result_table.setItem(row, 0, item_tcp)

        if item_tcp_corr is None:
            item_tcp_corr = QTableWidgetItem("0.00")
            self.result_table.setItem(row, 1, item_tcp_corr)   

        if item_tcp:
            try:
                value_tcp = float(item_tcp.text())
            except ValueError:
                value_tcp = 0.0
            value_tcp += delta
            item_tcp.setText(f"{value_tcp:.2f}")

        if item_tcp_corr:
            try:
                value_tcp_corr = float(item_tcp_corr.text())
            except ValueError:
                value_tcp = 0.0
            value_tcp_corr += delta
            item_tcp_corr.setText(f"{value_tcp_corr:.2f}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    with open("dark_theme.qss", "r") as f:
        app.setStyleSheet(f.read())

    window = MGDApp()
    #window.showMaximized()
    window.show()

    sys.exit(app.exec_())