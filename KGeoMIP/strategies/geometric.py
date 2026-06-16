import heapq
import time
import numpy as np
from typing import List, Dict, Tuple

from KGeoMIP.models.base.sia import SIA
from KGeoMIP.models.core.solution import Solution
from KGeoMIP.middlewares.slogger import SafeLogger
from KGeoMIP.middlewares.profile import gestor_perfilado, profile
from KGeoMIP.funcs.iit import emd_efecto, ABECEDARY
from KGeoMIP.funcs.format import fmt_biparticion_q

from KGeoMIP.constants.base import (
    ACTUAL,
    EFFECT as EFECTO,
    TYPE_TAG,
    NET_LABEL,
    STR_ZERO,
    COLS_IDX,
)

from KGeoMIP.constants.models import (
    GEOMETRIC_ANALYSIS_TAG,
    GEOMETRIC_LABEL,
    GEOMETRIC_STRAREGY_TAG,
)


class GeometricSIA(SIA):
    def __init__(self, tpm: np.ndarray):
        super().__init__(tpm)
        from KGeoMIP.models.base.application import aplicacion
        gestor_perfilado.start_session(
            f"{NET_LABEL}{len(tpm[COLS_IDX])}{aplicacion.pagina_red_muestra}"
        )
        self.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        self.logger = SafeLogger(GEOMETRIC_STRAREGY_TAG)
        self.tabla_transiciones: dict = {}
        self.vertices: set[tuple]
        self.tabla: dict[int, list[tuple[int, int]]] = {}
        self.memoria_particiones: dict[tuple[int, int], tuple[float, float]] = {}

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray = None,
    ):
        """vamos a hacer que vaya desde el estado inicial hasta el final, bit a bit diferente, llenando la tabla primero para distancias hamming 1 hasta n, con n la cantidad de bits que cambian del estado inicial al final. para esto podemos usar una tabla de transiciones, donde cada fila es un estado y cada columna es un bit. la tabla de transiciones se llena con los estados que se pueden alcanzar desde el estado inicial, y luego se va llenando la tabla de distancias hamming. para esto vamos a usar una lista de listas, donde cada lista es una fila de la tabla de transiciones. la primera fila es el estado inicial, y las siguientes filas son los estados alcanzables desde el estado inicial. la última fila es el estado final.
        paso a paso
        1. cargar la matriz, pasar a ncubos
        2. condicionar
        3. obtener los bits que cambian entre el estado inicial y el final
        4. obener vecinos del estado final que van hacia el estado inicial y calcular el costo de la transicion.
        5. para cada vecino, obtener los vecinos que van hacia el estado inicial y calcular el costo de la transicion.
        6. repetir hasta llegar al estado inicial.


        nota: intentar llenar la tabla desde el estado final hacia atras, pues al contrario habra dependencia de los valores de la tabla de los estados que van en camino hacia el estado final
        """
        if tpm is None:
            tpm = self.tpm

        N = tpm.shape[1]
        estado_inicial = "1" + ("0" * (N - 1))

        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        futuro = tuple(
            (EFECTO, efecto) for efecto in self.sia_subsistema.indices_ncubos
        )
        presente = tuple(
            (ACTUAL, actual) for actual in self.sia_subsistema.dims_ncubos
        )

        self._flat_data = []
        for idx, ncubo in enumerate(self.sia_subsistema.ncubos):
            # garantías: ncubo.data.shape == (2,2,...,2)
            # np.ravel() lo aplana. El orden ‘C’ equivale 
            # a little-endian si tus tuples están invertidas.
            self._flat_data.append(ncubo.data.ravel())

        self.vertices = set(presente + futuro)
        dims = self.sia_subsistema.dims_ncubos
        self.estado_inicial = self.sia_subsistema.estado_inicial[dims]
        self.estado_final = 1 - self.estado_inicial
        mip = self.find_mip()
        
        fmt_mip = fmt_biparticion_q(list(mip), self.nodes_complement(mip))

        return Solution(
            estrategia=GEOMETRIC_LABEL,
            perdida=self.memoria_particiones[mip][0],
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=self.memoria_particiones[mip][1],
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )
    
    def nodes_complement(self, nodes: list[tuple[int, int]]):
        return list(set(self.vertices) - set(nodes))
    
    def find_mip(self):
        """
        Implementa el algoritmo para encontrar la bipartición óptima
        utilizando el enfoque geométrico-topológico.
        """
        self.sia_logger.critic("empieza.")
        estado_inicial = self.estado_inicial
        estado_final = self.estado_final
        self.idx_ncubos = list(range(len(self.sia_subsistema.indices_ncubos)))
        self.caminos: Dict[int, List[List[int]]] = {0: [estado_inicial.tolist()]}
        self.tabla_transiciones[tuple(self.caminos[0][0]), tuple(self.caminos[0][0])] = [0.0 for _ in range(len(self.sia_subsistema.indices_ncubos))]
        for nivel in range(1, len(estado_inicial) + 1):
            self.calcular_costos_nivel(estado_final, nivel)
        candidatos = self.identificar_particiones_optimas()
        for idx, (presentes, futuros) in enumerate(candidatos):
            presentes = self.sia_subsistema.dims_ncubos[presentes]
            futuros = self.sia_subsistema.indices_ncubos[futuros]
            dist = self.sia_subsistema.bipartir(futuros, presentes).distribucion_marginal()
            emd = emd_efecto(dist, self.sia_dists_marginales)
            key = [(0, nodo) for nodo in presentes]
            key.extend([(1, nodo) for nodo in futuros])
            self.memoria_particiones[tuple(key)] = (emd, dist)
        return min(
            self.memoria_particiones, key=lambda k: self.memoria_particiones[k][0]
        )
    
    def calcular_costos_nivel(self, estado_final: np.ndarray, nivel):
        n = len(estado_final)      
        visitados: set[tuple] = set()
        self.caminos[nivel] = []
        for estado_anterior in self.caminos[nivel - 1]:
            estado_actual = np.array(estado_anterior)
            for i in range(n):
                if estado_actual[i] != estado_final[i]:
                    nuevo_estado = estado_actual.copy()
                    nuevo_estado[i] = estado_final[i]
                    nuevo_estado_tuple = tuple(nuevo_estado)
                    if nuevo_estado_tuple not in visitados:
                        self.caminos[nivel].append(nuevo_estado.tolist())
                        self.calcular_costo(self.caminos[0][0], nuevo_estado.tolist(), self.idx_ncubos)
                        visitados.add(nuevo_estado_tuple)

    def calcular_costo(self, estado_inicial: tuple, estado_final: tuple, ncubos: list[int]):
        key = tuple(estado_inicial), tuple(estado_final)
        if key not in self.tabla_transiciones:
            self.tabla_transiciones[key] = [None] * len(self.sia_subsistema.indices_ncubos)
        distancia_hamming = self.hamming(estado_inicial, estado_final)
        factor = 1 / (2 ** distancia_hamming)

        estado_ini_int = int("".join(map(str, estado_inicial[::-1])), 2)
        estado_fin_int = int("".join(map(str, estado_final[::-1])), 2)

        diffs = np.abs(
            np.array([flat[estado_ini_int] for flat in self._flat_data])
            - np.array([flat[estado_fin_int] for flat in self._flat_data])
        )
        self.tabla_transiciones[key] = diffs.tolist()
        
        if distancia_hamming > 1:
            for i in range(len(estado_inicial)):
                if estado_inicial[i] != estado_final[i]:
                    nuevo_estado = estado_final.copy()
                    nuevo_estado[i] = estado_inicial[i]
                    nuevo_estado_tuple = tuple(nuevo_estado)
                    temp_key = tuple(estado_inicial), nuevo_estado_tuple
                    for n in ncubos:
                        self.tabla_transiciones[key][n] = self.tabla_transiciones[key][n] + self.tabla_transiciones[temp_key][n]
        tmp = []
        for i, n in enumerate(self.tabla_transiciones[key]):
            if n is not None:
                tmp.append(factor * n)
            else:
                tmp.append(n)
        self.tabla_transiciones[key] = tmp

    def identificar_particiones_optimas(self):
        key = tuple(self.caminos[0][0]), tuple(self.estado_final)
        costos: list = self.tabla_transiciones[key]
        candidatos = []
        n_vars = len(costos)
        for idx in range(n_vars):
            presentes = [i for i in range(len(self.estado_final))]
            futuros = [i for i in range(n_vars) if i != idx]
            candidatos.append([presentes, futuros])
        es_par = len(self.caminos) % 2 == 0
        if es_par:
            mitad = len(self.caminos) // 2
        else:
            mitad = (len(self.caminos) // 2) + 1
        for nivel in range(1, mitad):
            costo_candidato_nivel = 1e5
            presentes_nivel = []
            futuros_nivel = []
            for estado in self.caminos[nivel]:
                costo_candidato = 0
                presentes = []
                futuros = []
                actual = self.tabla_transiciones.get((tuple(self.caminos[0][0]), tuple(estado)), None)
                estado_complementario = (1 - np.array(estado)).tolist()
                complementario = self.tabla_transiciones.get((tuple(self.caminos[0][0]), tuple(estado_complementario)), None)
                for idx, i in enumerate(estado):
                    if i == self.caminos[0][0][idx]:
                        presentes.append(idx)
                for idx, _ in enumerate(self.idx_ncubos):
                    if actual[idx] <= complementario[idx]:
                        futuros.append(idx)
                        costo_candidato += actual[idx]
                    else:
                        costo_candidato += complementario[idx]
                if costo_candidato < costo_candidato_nivel:
                    costo_candidato_nivel = costo_candidato
                    presentes_nivel = presentes
                    futuros_nivel = futuros
            candidatos.append([presentes_nivel, futuros_nivel])
        return candidatos

    def find_mip_for_subsystem(self, sub_sys):
        """
        Calcula candidatos de bipartición óptimos para un sub-sistema utilizando
        el enfoque geométrico-topológico.
        """
        dims = sub_sys.dims_ncubos
        estado_inicial = sub_sys.estado_inicial[dims]
        estado_final = 1 - estado_inicial
        
        idx_ncubos = list(range(len(sub_sys.indices_ncubos)))
        caminos: Dict[int, List[List[int]]] = {0: [estado_inicial.tolist()]}
        tabla_transiciones = {}
        tabla_transiciones[tuple(caminos[0][0]), tuple(caminos[0][0])] = [0.0 for _ in range(len(sub_sys.indices_ncubos))]
        
        flat_data = [ncubo.data.ravel() for ncubo in sub_sys.ncubos]
        
        def calcular_costo_local(est_ini: tuple, est_fin: tuple, ncubos: list[int]):
            key = tuple(est_ini), tuple(est_fin)
            if key not in tabla_transiciones:
                tabla_transiciones[key] = [None] * len(sub_sys.indices_ncubos)
            distancia_hamming = sum(x != y for x, y in zip(est_ini, est_fin))
            factor = 1 / (2 ** distancia_hamming)

            estado_ini_int = int("".join(map(str, est_ini[::-1])), 2)
            estado_fin_int = int("".join(map(str, est_fin[::-1])), 2)

            diffs = np.abs(
                np.array([flat[estado_ini_int] for flat in flat_data])
                - np.array([flat[estado_fin_int] for flat in flat_data])
            )
            tabla_transiciones[key] = diffs.tolist()
            
            if distancia_hamming > 1:
                for i in range(len(est_ini)):
                    if est_ini[i] != est_fin[i]:
                        nuevo_estado = list(est_fin)
                        nuevo_estado[i] = est_ini[i]
                        nuevo_estado_tuple = tuple(nuevo_estado)
                        temp_key = tuple(est_ini), nuevo_estado_tuple
                        for n in ncubos:
                            tabla_transiciones[key][n] = tabla_transiciones[key][n] + tabla_transiciones[temp_key][n]
            tmp = []
            for i, n in enumerate(tabla_transiciones[key]):
                if n is not None:
                    tmp.append(factor * n)
                else:
                    tmp.append(n)
            tabla_transiciones[key] = tmp

        def calcular_costos_nivel_local(est_final: np.ndarray, nivel):
            n = len(est_final)      
            visitados: set[tuple] = set()
            caminos[nivel] = []
            for estado_anterior in caminos[nivel - 1]:
                estado_actual = np.array(estado_anterior)
                for i in range(n):
                    if estado_actual[i] != est_final[i]:
                        nuevo_estado = estado_actual.copy()
                        nuevo_estado[i] = est_final[i]
                        nuevo_estado_tuple = tuple(nuevo_estado)
                        if nuevo_estado_tuple not in visitados:
                            caminos[nivel].append(nuevo_estado.tolist())
                            calcular_costo_local(caminos[0][0], nuevo_estado.tolist(), idx_ncubos)
                            visitados.add(nuevo_estado_tuple)

        for nivel in range(1, len(estado_inicial) + 1):
            calcular_costos_nivel_local(estado_final, nivel)
            
        key = tuple(caminos[0][0]), tuple(estado_final)
        costos: list = tabla_transiciones[key]
        candidatos = []
        n_vars = len(costos)
        for idx in range(n_vars):
            presentes = [i for i in range(len(estado_final))]
            futuros = [i for i in range(n_vars) if i != idx]
            candidatos.append([presentes, futuros])
        es_par = len(caminos) % 2 == 0
        if es_par:
            mitad = len(caminos) // 2
        else:
            mitad = (len(caminos) // 2) + 1
        for nivel in range(1, mitad):
            costo_candidato_nivel = 1e5
            presentes_nivel = []
            futuros_nivel = []
            for estado in caminos[nivel]:
                costo_candidato = 0
                presentes = []
                futuros = []
                actual = tabla_transiciones.get((tuple(caminos[0][0]), tuple(estado)), None)
                estado_complementario = (1 - np.array(estado)).tolist()
                complementario = tabla_transiciones.get((tuple(caminos[0][0]), tuple(estado_complementario)), None)
                for idx, i in enumerate(estado):
                    if i == caminos[0][0][idx]:
                        presentes.append(idx)
                for idx, _ in enumerate(idx_ncubos):
                    if actual[idx] <= complementario[idx]:
                        futuros.append(idx)
                        costo_candidato += actual[idx]
                    else:
                        costo_candidato += complementario[idx]
                if costo_candidato < costo_candidato_nivel:
                    costo_candidato_nivel = costo_candidato
                    presentes_nivel = presentes
                    futuros_nivel = futuros
            candidatos.append([presentes_nivel, futuros_nivel])
            
        candidatos_particiones = []
        for presentes_sub, futuros_sub in candidatos:
            presentes_nodos = [sub_sys.dims_ncubos[p] for p in presentes_sub]
            futuros_nodos = [sub_sys.indices_ncubos[f] for f in futuros_sub]
            g1 = [(0, p) for p in presentes_nodos] + [(1, f) for f in futuros_nodos]
            candidatos_particiones.append(g1)
            
        return candidatos_particiones

    def aplicar_estrategia_k(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
        k: int,
    ):
        from KGeoMIP.funcs.iit import emd_efecto, ABECEDARY
        from KGeoMIP.models.core.solution import Solution
        from KGeoMIP.constants.models import GEOMETRIC_LABEL
        import time

        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)
        
        futuro = tuple((1, idx_efecto) for idx_efecto in self.sia_subsistema.indices_ncubos)
        presente = tuple((0, idx_actual) for idx_actual in self.sia_subsistema.dims_ncubos)
        
        self.vertices = set(presente + futuro)
        
        grupos_actuales = [list(presente + futuro)]
        
        def group_to_side(grupo):
            alc = [idx for t, idx in grupo if t == 1]
            mec = [idx for t, idx in grupo if t == 0]
            return (np.array(sorted(alc), dtype=np.int8), np.array(sorted(mec), dtype=np.int8))

        def fmt_grupo(g):
            alc = sorted([idx for t, idx in g if t == 1])
            mec = sorted([idx for t, idx in g if t == 0])
            t1_str = ",".join(ABECEDARY[i] for i in alc) if alc else "∅"
            t_str = ",".join(ABECEDARY[i].lower() for i in mec) if mec else "∅"
            return f"{{ {t1_str}_t+1, {t_str}_t }}"

        # 1. Biparticiones sucesivas hasta alcanzar k particiones
        while len(grupos_actuales) < k:
            best_split_idx = -1
            best_sub_grupos = None
            best_nueva_perdida = float('inf')
            
            for idx_g, grupo in enumerate(grupos_actuales):
                if len(grupo) <= 1:
                    continue
                
                # Generate candidates for splitting 'grupo'
                alc_grupo = [idx for t, idx in grupo if t == 1]
                mec_grupo = [idx for t, idx in grupo if t == 0]
                
                candidatos_g1 = []
                if alc_grupo and mec_grupo:
                    alcance_to_remove = np.setdiff1d(self.sia_subsistema.indices_ncubos, alc_grupo)
                    mecanismo_to_remove = np.setdiff1d(self.sia_subsistema.dims_ncubos, mec_grupo)
                    restricted_subsystem = self.sia_subsistema.substraer(alcance_to_remove, mecanismo_to_remove)
                    
                    candidatos_g1 = self.find_mip_for_subsystem(restricted_subsystem)
                else:
                    for node in grupo:
                        candidatos_g1.append([node])
                    if len(grupo) > 3:
                        mid = len(grupo) // 2
                        candidatos_g1.append(grupo[:mid])
                
                for g1 in candidatos_g1:
                    g2 = list(set(grupo) - set(g1))
                    if not g1 or not g2:
                        continue
                    
                    # Construir los k grupos temporales
                    temp_grupos = list(grupos_actuales)
                    temp_grupos.pop(idx_g)
                    temp_grupos.extend([g1, g2])
                    
                    # Obtener lados y armar el sistema particionado
                    sides = [group_to_side(g) for g in temp_grupos]
                    sistema_k = self.sia_subsistema.k_partir(sides)
                    
                    nueva_perdida = emd_efecto(sistema_k.distribucion_marginal(), self.sia_dists_marginales)
                    
                    if nueva_perdida < best_nueva_perdida:
                        best_nueva_perdida = nueva_perdida
                        best_sub_grupos = (g1, g2)
                        best_split_idx = idx_g
                        
            if best_split_idx == -1:
                break # No further subdivisions possible
                
            grupos_actuales.pop(best_split_idx)
            grupos_actuales.extend(best_sub_grupos)
            
        sides = [group_to_side(g) for g in grupos_actuales]
        sistema_final = self.sia_subsistema.k_partir(sides)
        perdida_final = emd_efecto(sistema_final.distribucion_marginal(), self.sia_dists_marginales)
        
        particion_str = " ⊗ ".join(fmt_grupo(g) for g in grupos_actuales)
        
        return Solution(
            estrategia=GEOMETRIC_LABEL,
            perdida=perdida_final,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=sistema_final.distribucion_marginal(),
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=particion_str,
        )

    def hamming(self, a: List[int], b: List[int]) -> int:
        return sum(x != y for x, y in zip(a, b))

