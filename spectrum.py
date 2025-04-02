
import pygame
import pyaudio
import numpy as np
import math
import colorsys
import win32gui
import win32con
import win32api

# Initialisation de Pygame
pygame.init()
WIDTH, HEIGHT = 300, 300

screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.NOFRAME | pygame.SRCALPHA)
clock = pygame.time.Clock()

# Ajouter après l'initialisation de la fenêtre
hwnd = pygame.display.get_wm_info()["window"]

win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, 
                      win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED)
win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(0,0,0), 0, win32con.LWA_COLORKEY)

# Initialisation audio
CHUNK = 1024  
RATE = 44100  
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Paramètres du cercle
base_radius = 50
num_points = 800  # Nombre de points pour former le cercle
color_hue = 0  

running = True
while running:
    screen.fill((0, 0, 0, 0))  

    # Capture du son en temps réel
    data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
    spectrum = np.fft.fft(data)  # Transformée de Fourier pour obtenir le spectre
    amplitude = np.max(np.abs(spectrum))  # Intensité du son

    # Définition de l'amplitude des vagues (plus y'a de son, plus ça ondule)
    wave_strength = (amplitude / 8000)  # Ajuster l'intensité des vagues

    # Changer la couleur dynamiquement
    color_hue += 0.001  # Vitesse de changement de couleur
    if color_hue > 1:
        color_hue = 0
    r, g, b = colorsys.hsv_to_rgb(color_hue, 1, 1)  
    color = (int(r * 255), int(g * 255), int(b * 255))

    # Génération des points du cercle déformé
    points = []
    for i in range(num_points):
        # Cercle 1
        angle = (i / num_points) * 2 * math.pi  # Angle de chaque point
        # Ondulations dynamiques, nombre de waves (8) et vitesse de rotation pygame.time.get_ticks() * 0.005 et apparation des waves
        noise = wave_strength * math.sin(8 * angle + pygame.time.get_ticks() * 0.001)  # + pygame.time.get_ticks() * 0.001
        radius = base_radius + noise  # Modifier le rayon avec le bruit
        x = WIDTH // 2 + int(radius * math.cos(angle))
        y = HEIGHT // 2 + int(radius * math.sin(angle))
        points.append((x, y))

        # Cercle 2
        angle = (i / num_points) * 2 * math.pi
        noise = wave_strength * math.sin(12 * angle + pygame.time.get_ticks() * 0.005)
        radius = base_radius + noise
        x = WIDTH // 2 + int(radius * math.cos(angle))
        y = HEIGHT // 2 + int(radius * math.sin(angle))
        points.append((x, y))

        # Cercle 3
        angle = (i / num_points) * 2 * math.pi
        noise = wave_strength * math.sin(6 * angle - pygame.time.get_ticks() * 0.001) #+ pygame.time.get_ticks() * 0.001
        radius = base_radius + noise
        x = WIDTH // 2 + int(radius * math.cos(angle))
        y = HEIGHT // 2 + int(radius * math.sin(angle))
        points.append((x, y))

    # Dessiner le cercle déformé
    pygame.draw.polygon(screen, color, points, 2)

    # In the event loop, add:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            win32gui.ReleaseCapture()
            win32gui.SendMessage(hwnd, win32con.WM_NCLBUTTONDOWN, win32con.HTCAPTION, 0)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE: 
                running = False

    pygame.display.flip()
    clock.tick(60)  

pygame.quit()
