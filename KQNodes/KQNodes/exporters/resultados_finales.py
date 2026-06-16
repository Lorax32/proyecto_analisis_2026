from __future__ import annotations

import logging
import multiprocessing
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from KQNodes.KQNodes.constants.base import COLON_DELIM, CSV_EXTENSION, PATH_SAMPLES
from KQNodes.KQNodes.constants.models import (
    GEOMETRIC_STRAREGY_TAG,
    QNODES_STRAREGY_TAG,
    SIA_PREPARATION_TAG,
)
from KQNodes.KQNodes.models.core.solution import Solution
from KGeoMIP.strategies.geometric import GeometricSIA
from KQNodes.KQNodes.strategies.q_nodes import KQNodes

SAMPLE_PATTERN = re.compile(rf"^N(\d+)([A-Z])\.{CSV_EXTENSION}$")
RESULTADOS_FILENAME = "resultadosFinales.xlsx"
SUBPROCESO_TIMEOUT_S = 120
LOGGERS_SILENCIADOS = (
    SIA_PREPARATION_TAG,
    GEOMETRIC_STRAREGY_TAG,
    QNODES_STRAREGY_TAG,
)


def _ruta_salida_por_defecto() -> Path:
    # __file__ = <project>/KQNodes/KQNodes/exporters/resultados_finales.py
    # parents: [0]=exporters, [1]=KQNodes, [2]=KQNodes(outer), [3]=project root
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "KQNodes" / "results" / RESULTADOS_FILENAME


def _caso_principal(n: int) -> dict[str, str]:
    """Mismo subsistema que usa ``main.py`` con el sistema completo."""
    mascara = "1" * n
    return {
        "estado_inicial": "1" + ("0" * (n - 1)),
        "condiciones": mascara,
        "alcance": mascara,
        "mecanismo": mascara,
    }


def _silenciar_loggers_hijos() -> None:
    for nombre in LOGGERS_SILENCIADOS:
        logger = logging.getLogger(nombre)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.CRITICAL + 10)


def _subproceso_estrategia(
    nombre_estrategia: str,
    tpm: np.ndarray,
    caso: dict[str, str],
    k: int,
    cola: multiprocessing.Queue,
) -> None:
    _silenciar_loggers_hijos()
    try:
        estrategia_cls = KQNodes if nombre_estrategia == "KQNodes" else GeometricSIA
        analizador = estrategia_cls(tpm)
        _silenciar_loggers_hijos()
        inicio = time.perf_counter()
        solucion = analizador.aplicar_estrategia_k(
            caso["estado_inicial"],
            caso["condiciones"],
            caso["alcance"],
            caso["mecanismo"],
            k=k,
        )
        cola.put(
            (
                "ok",
                {
                    "estrategia": solucion.estrategia,
                    "perdida": float(solucion.perdida),
                    "particion": solucion.particion,
                    "tiempo": time.perf_counter() - inicio,
                },
            )
        )
    except RecursionError:
        cola.put(("err", "RecursionError: maximum recursion depth exceeded"))
    except Exception as exc:
        cola.put(("err", _mensaje_error_seguro(exc)))


def _mensaje_error_seguro(exc: BaseException) -> str:
    nombre = type(exc).__name__
    if isinstance(exc, RecursionError):
        return "RecursionError: maximum recursion depth exceeded"
    try:
        return f"{nombre}: {exc}"
    except Exception:
        return nombre


class ExportadorResultadosFinales:
    """
    Ejecuta KQNodes y Geometric sobre todas las redes en ``src/.samples/``
    con el mismo caso de ``main.py`` (sistema completo) y guarda un Excel
    comparativo en ``GeoMIP/results/resultadosFinales.xlsx``.
    """

    def __init__(self, ruta_muestras: Path | None = None, ruta_salida: Path | None = None):
        self.ruta_muestras = Path(ruta_muestras or PATH_SAMPLES)
        self.ruta_salida = Path(ruta_salida or _ruta_salida_por_defecto())

    def ejecutar(
        self,
        n_inicio: int = 2,
        n_fin: int = 5,
        k_inicio: int = 2,
        k_fin: int = 5,
    ) -> Path:
        if n_inicio > n_fin:
            raise ValueError("n_inicio no puede ser mayor que n_fin")
        if k_inicio > k_fin:
            raise ValueError("k_inicio no puede ser mayor que k_fin")
        if k_inicio < 2:
            raise ValueError("k_inicio debe ser al menos 2")

        muestras = self._listar_muestras(n_inicio, n_fin)
        if not muestras:
            raise FileNotFoundError(
                f"No hay archivos N*.{CSV_EXTENSION} en {self.ruta_muestras} "
                f"con N entre {n_inicio} y {n_fin}"
            )

        filas: list[dict] = []
        iteracion = 0
        interrumpido = False

        print(f"[resultadosFinales] Salida: {self.ruta_salida.resolve()}")

        try:
            with self._silenciar_logs():
                for ruta_red, n, variante in muestras:
                    tpm = np.genfromtxt(ruta_red, delimiter=COLON_DELIM)
                    caso = _caso_principal(n)
                    print(f"\n[resultadosFinales] Red {ruta_red.name} (N={n})")

                    for k in range(k_inicio, k_fin + 1):
                        iteracion += 1
                        try:
                            fila = self._evaluar_caso(
                                iteracion=iteracion,
                                red=ruta_red.stem,
                                n=n,
                                variante=variante,
                                k=k,
                                tpm=tpm,
                                caso=caso,
                            )
                        except Exception as exc:
                            fila = self._fila_error(
                                iteracion=iteracion,
                                red=ruta_red.stem,
                                n=n,
                                variante=variante,
                                k=k,
                                error=_mensaje_error_seguro(exc),
                            )

                        filas.append(fila)
                        print(
                            f"  [{iteracion}] k={k} "
                            f"KQNodes={fila['Pérdida KQNodes']} "
                            f"Geometric={fila['Pérdida Geometric']}"
                        )
        except KeyboardInterrupt:
            interrumpido = True
            print("\n[resultadosFinales] Ejecución interrumpida. Guardando resultados parciales...")
        finally:
            if filas:
                self._guardar_excel(filas)
                estado = "parciales" if interrumpido else "finales"
                print(
                    f"\n[resultadosFinales] {len(filas)} filas {estado} guardadas en "
                    f"{self.ruta_salida.resolve()}"
                )
            else:
                print("\n[resultadosFinales] No hay filas para guardar.")

        return self.ruta_salida

    def _guardar_excel(self, filas: list[dict]) -> None:
        df = pd.DataFrame(filas, columns=self._columnas())
        self.ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(self.ruta_salida, index=False)

    def _listar_muestras(self, n_inicio: int, n_fin: int) -> list[tuple[Path, int, str]]:
        muestras: list[tuple[Path, int, str]] = []
        for archivo in sorted(self.ruta_muestras.glob(f"N*.{CSV_EXTENSION}")):
            coincidencia = SAMPLE_PATTERN.match(archivo.name)
            if not coincidencia:
                continue
            n = int(coincidencia.group(1))
            if n_inicio <= n <= n_fin:
                muestras.append((archivo, n, coincidencia.group(2)))
        return sorted(muestras, key=lambda item: (item[1], item[2]))

    def _evaluar_caso(
        self,
        *,
        iteracion: int,
        red: str,
        n: int,
        variante: str,
        k: int,
        tpm: np.ndarray,
        caso: dict[str, str],
    ) -> dict:
        fila = self._fila_base(iteracion, red, n, variante, k)

        sol_kq = self._ejecutar_estrategia("KQNodes", tpm, caso, k)
        sol_geo = self._ejecutar_estrategia("Geometric", tpm, caso, k)

        self._rellenar_resultado(fila, sol_kq, prefijo="KQNodes")
        self._rellenar_resultado(fila, sol_geo, prefijo="Geometric")
        self._calcular_comparativa(fila)
        return fila

    def _fila_base(
        self,
        iteracion: int,
        red: str,
        n: int,
        variante: str,
        k: int,
    ) -> dict:
        return {
            "Iteración": iteracion,
            "Red": red,
            "N": n,
            "Variante": variante,
            "k": k,
            "Partición KQNodes": None,
            "Pérdida KQNodes": None,
            "Tiempo KQNodes (s)": None,
            "Partición Geometric": None,
            "Pérdida Geometric": None,
            "Tiempo Geometric (s)": None,
            "Δ Pérdida": None,
            "Particiones coinciden": None,
            "Error KQNodes": "",
            "Error Geometric": "",
        }

    def _fila_error(
        self,
        *,
        iteracion: int,
        red: str,
        n: int,
        variante: str,
        k: int,
        error: str,
    ) -> dict:
        fila = self._fila_base(iteracion, red, n, variante, k)
        fila["Error KQNodes"] = error
        fila["Error Geometric"] = error
        return fila

    def _rellenar_resultado(self, fila: dict, resultado: Solution | str, *, prefijo: str) -> None:
        if isinstance(resultado, Solution):
            fila[f"Partición {prefijo}"] = resultado.particion
            fila[f"Pérdida {prefijo}"] = resultado.perdida
            fila[f"Tiempo {prefijo} (s)"] = resultado.tiempo_ejecucion
        else:
            fila[f"Error {prefijo}"] = resultado

    def _calcular_comparativa(self, fila: dict) -> None:
        if fila["Pérdida KQNodes"] is not None and fila["Pérdida Geometric"] is not None:
            fila["Δ Pérdida"] = fila["Pérdida Geometric"] - fila["Pérdida KQNodes"]
            fila["Particiones coinciden"] = (
                fila["Partición KQNodes"] == fila["Partición Geometric"]
            )

    def _ejecutar_estrategia(
        self,
        nombre_estrategia: str,
        tpm: np.ndarray,
        caso: dict[str, str],
        k: int,
    ) -> Solution | str:
        if sys.platform == "win32":
            return self._ejecutar_estrategia_subproceso(nombre_estrategia, tpm, caso, k)
        return self._ejecutar_estrategia_directa(nombre_estrategia, tpm, caso, k)

    def _ejecutar_estrategia_directa(
        self,
        nombre_estrategia: str,
        tpm: np.ndarray,
        caso: dict[str, str],
        k: int,
    ) -> Solution | str:
        try:
            estrategia_cls = KQNodes if nombre_estrategia == "KQNodes" else GeometricSIA
            analizador = estrategia_cls(tpm)
            inicio = time.perf_counter()
            solucion = analizador.aplicar_estrategia_k(
                caso["estado_inicial"],
                caso["condiciones"],
                caso["alcance"],
                caso["mecanismo"],
                k=k,
            )
            solucion.hablar = False
            solucion.tiempo_ejecucion = time.perf_counter() - inicio
            return solucion
        except RecursionError:
            return "RecursionError: maximum recursion depth exceeded"
        except Exception as exc:
            return _mensaje_error_seguro(exc)

    def _ejecutar_estrategia_subproceso(
        self,
        nombre_estrategia: str,
        tpm: np.ndarray,
        caso: dict[str, str],
        k: int,
    ) -> Solution | str:
        ctx = multiprocessing.get_context("spawn")
        cola: multiprocessing.Queue = ctx.Queue()
        proceso = ctx.Process(
            target=_subproceso_estrategia,
            args=(nombre_estrategia, tpm, caso, k, cola),
            daemon=True,
        )
        proceso.start()
        proceso.join(timeout=SUBPROCESO_TIMEOUT_S)

        if proceso.is_alive():
            proceso.terminate()
            proceso.join()
            return f"TimeoutError: excedió {SUBPROCESO_TIMEOUT_S}s"

        if cola.empty():
            return "RuntimeError: el subproceso terminó sin devolver resultado"

        estado, payload = cola.get()
        if estado == "ok":
            return Solution(
                estrategia=payload["estrategia"],
                perdida=payload["perdida"],
                distribucion_subsistema=np.array([]),
                distribucion_particion=np.array([]),
                particion=payload["particion"],
                tiempo_total=payload["tiempo"],
                quiere_hablar=False,
            )
        return payload

    @contextmanager
    def _silenciar_logs(self):
        niveles_previos: dict[str, int] = {}
        for nombre in LOGGERS_SILENCIADOS:
            logger = logging.getLogger(nombre)
            niveles_previos[nombre] = logger.level
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.setLevel(logging.CRITICAL + 10)
        try:
            yield
        finally:
            for nombre, nivel in niveles_previos.items():
                logging.getLogger(nombre).setLevel(nivel)

    @staticmethod
    def _columnas() -> list[str]:
        return [
            "Iteración",
            "Red",
            "N",
            "Variante",
            "k",
            "Partición KQNodes",
            "Pérdida KQNodes",
            "Tiempo KQNodes (s)",
            "Partición Geometric",
            "Pérdida Geometric",
            "Tiempo Geometric (s)",
            "Δ Pérdida",
            "Particiones coinciden",
            "Error KQNodes",
            "Error Geometric",
        ]
