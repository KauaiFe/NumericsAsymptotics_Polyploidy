"""Render the README equations as portable, self-contained SVG images."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EQUATIONS = {
    'corrected-radius': [r'$R_c\sim\frac{2\sigma}{3\upsilon}\sqrt{\frac{1-\phi}{2}}$'],
    'equilibria': [
        r'$y_s^*=\frac{\upsilon}{A}+O(\upsilon^2),\qquad y_u^*=\frac{1}{2}-\frac{\upsilon}{4A}+O(\upsilon^2)$',
    ],
    'wave-speed': [r'$c_0=\frac{\sigma\sqrt{k}}{2}(1+y_s^*-2y_u^*),\qquad k=2A-\upsilon$'],
    'balance': [r'$1+y_s^*-2y_u^*=\frac{\upsilon}{A}+\frac{\upsilon}{2A}+O(\upsilon^2)=\frac{3\upsilon}{2A}+O(\upsilon^2)$'],
    'speed-radius': [r'$c_0=\frac{3\sigma\upsilon}{2\sqrt{2A}}+O(\upsilon^2),\qquad R_c\approx\frac{D}{c_0},\quad D=\frac{\sigma^2}{2}$'],
    'model': [r'$\frac{\partial y}{\partial t}=\frac{\sigma^2}{2}\left(\frac{\partial^2y}{\partial r^2}+\frac{1}{r}\frac{\partial y}{\partial r}\right)+f(y)$'],
    'reaction': [r'$f(y)=\upsilon+(\phi-2\upsilon-1)y+3(1-\phi)y^2+(2\phi+\upsilon-2)y^3$'],
}


def main():
    destination = Path(__file__).resolve().parent
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'svg.fonttype': 'path', 'svg.hashsalt': 'polyploidy-equations', 'font.size': 17})
    for name, lines in EQUATIONS.items():
        fig = plt.figure(figsize=(10, 0.8 * len(lines)), facecolor='white')
        for index, equation in enumerate(lines):
            fig.text(0.02, (len(lines) - index - 0.5) / len(lines), equation, va='center', color='#111111')
        # Save only the text extent, leaving a small white margin for dark themes.
        fig.savefig(destination / f'equation-{name}.svg', bbox_inches='tight', pad_inches=0.16,
                    facecolor='white', metadata={'Date': None})
        plt.close(fig)
    print(f'Rendered {len(EQUATIONS)} equation images.')


if __name__ == '__main__':
    main()
