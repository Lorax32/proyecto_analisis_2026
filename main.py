from KQNodes.KQNodes.controllers.manager import Manager
from KQNodes.KQNodes.models.base.application import aplicacion as aplicacion_kq
from KQNodes.BruteForce.models.base.application import aplicacion as aplicacion_bf
from KGeoMIP.models.base.application import aplicacion as aplicacion_geo

# Estrategias
from KQNodes.KQNodes.strategies.q_nodes import KQNodes
from KQNodes.BruteForce.strategies.force import BruteForce
from KGeoMIP.strategies.geometric import GeometricSIA

# Exportador de resultados masivos
from KQNodes.KQNodes.exporters.resultados_finales import ExportadorResultadosFinales


def _sync_version(version: str) -> None:
    aplicacion_kq.set_pagina_red_muestra(version)
    aplicacion_bf.set_pagina_red_muestra(version)
    aplicacion_geo.set_pagina_red_muestra(version)


def ejecutar_estrategia(n, k, version, opcion):
    estado_inicial = "1" + ("0" * (n - 1))
    condiciones = "1" * n
    alcance = "1" * n
    mecanismo = "1" * n

    _sync_version(version)

    gestor_redes = Manager(estado_inicial)

    print("\n===================================")
    print(f"Archivo TPM: {gestor_redes.tpm_filename}")
    print(f"N = {n}")
    print(f"K = {k}")
    print(f"Versión = {version}")
    print("===================================\n")

    # Verificar si existe la red solicitada
    nombre_archivo = f"N{n}{version}.csv"
    ruta_red = gestor_redes.ruta_base / nombre_archivo

    if not ruta_red.exists():
        print(
            f"La red {nombre_archivo} no existe. "
            f"Generando automáticamente..."
        )

        gestor_redes.generar_red(n)

        print(f"Red {nombre_archivo} generada correctamente.\n")

    # Cargar la red (existente o recién generada)
    mpt = gestor_redes.cargar_red()

    if opcion == 1:
        print("Ejecutando Brute Force...\n")

        analizador = BruteForce(mpt)

        resultado = analizador.aplicar_estrategia(
            estado_inicial,
            condiciones,
            alcance,
            mecanismo,
        )

    elif opcion == 2:
        print("Ejecutando KQNodes...\n")

        analizador = KQNodes(mpt)

        resultado = analizador.aplicar_estrategia_k(
            estado_inicial,
            condiciones,
            alcance,
            mecanismo,
            k=k,
        )

    elif opcion == 3:
        print("Ejecutando Geometric...\n")

        analizador = GeometricSIA(mpt)

        resultado = analizador.aplicar_estrategia_k(
            estado_inicial,
            condiciones,
            alcance,
            mecanismo,
            k=k,
        )

    else:
        print("Opción inválida.")
        return

    print("\nResultado:")
    print(resultado)
    print()


def generar_resultados_masivos():
    """
    Ejecuta KQNodes y KGeoMIP de forma masiva iterando sobre rangos de N y K,
    almacenando los resultados en KQNodes/results/resultadosFinales.xlsx.
    """
    print("\n===== GENERACIÓN MASIVA DE RESULTADOS =====")
    print("Se ejecutarán las estrategias KQNodes y KGeoMIP")
    print("para todas las combinaciones de N y K indicadas.")
    print("Los resultados se guardarán en:")
    print("  KQNodes/results/resultadosFinales.xlsx")
    print("===========================================\n")

    try:
        n_inicio = int(input("Ingrese N inicial (mínimo 2): "))
        n_fin = int(input("Ingrese N final: "))
        k_inicio = int(input("Ingrese K inicial (mínimo 2): "))
        k_fin = int(input("Ingrese K final: "))
    except ValueError:
        print("Todos los valores deben ser números enteros.")
        return

    if n_inicio < 2:
        print("N inicial debe ser mayor o igual a 2.")
        return

    if n_inicio > n_fin:
        print("N inicial no puede ser mayor que N final.")
        return

    if k_inicio < 2:
        print("K inicial debe ser mayor o igual a 2.")
        return

    if k_inicio > k_fin:
        print("K inicial no puede ser mayor que K final.")
        return

    confirmar = input(
        f"\n¿Confirma ejecutar para N=[{n_inicio}..{n_fin}] "
        f"y K=[{k_inicio}..{k_fin}]? (s/n): "
    ).lower()

    if confirmar != "s":
        print("Operación cancelada.")
        return

    print("\nIniciando generación de resultados...")
    print("(Esto puede tardar dependiendo del rango seleccionado)\n")

    try:
        exportador = ExportadorResultadosFinales()
        ruta_salida = exportador.ejecutar(
            n_inicio=n_inicio,
            n_fin=n_fin,
            k_inicio=k_inicio,
            k_fin=k_fin,
        )
        print(f"\n✓ Resultados guardados correctamente en:")
        print(f"  {ruta_salida}")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
    except ValueError as e:
        print(f"\n✗ Error de validación: {e}")
    except Exception as e:
        print(f"\n✗ Error inesperado: {e}")


def mostrar_menu():
    while True:
        print("\n========= MENÚ PRINCIPAL =========")
        print("1. Brute Force")
        print("2. KQNodes")
        print("3. Geometric")
        print("4. Generar resultados masivos (KQNodes vs KGeoMIP)")
        print("0. Salir")
        print("==================================")

        try:
            opcion = int(input("Seleccione una opción: "))
        except ValueError:
            print("Ingrese un número válido.")
            continue

        if opcion == 0:
            print("Saliendo...")
            break

        if opcion not in [1, 2, 3, 4]:
            print("Opción inválida.")
            continue

        if opcion == 4:
            generar_resultados_masivos()

            repetir = input(
                "\n¿Desea realizar otra operación? (s/n): "
            ).lower()

            if repetir != "s":
                print("Finalizando programa...")
                break

            continue

        try:
            n = int(input("Ingrese N: "))
            k = int(input("Ingrese K: "))
        except ValueError:
            print("N y K deben ser números enteros.")
            continue

        if n < 2:
            print("N debe ser mayor o igual a 2.")
            continue

        if k < 1:
            print("K debe ser mayor o igual a 1.")
            continue

        if k > n:
            print("K no puede ser mayor que N.")
            continue

        version = input(
            "Ingrese la versión (A/B/C/...): "
        ).strip().upper()

        ejecutar_estrategia(
            n=n,
            k=k,
            version=version,
            opcion=opcion,
        )

        repetir = input(
            "\n¿Desea ejecutar otra estrategia? (s/n): "
        ).lower()

        if repetir != "s":
            print("Finalizando programa...")
            break


def main():
    """Inicialización del aplicativo"""

    aplicacion_kq.activar_profiling()
    aplicacion_bf.activar_profiling()
    aplicacion_geo.activar_profiling()

    mostrar_menu()


if __name__ == "__main__":
    main()