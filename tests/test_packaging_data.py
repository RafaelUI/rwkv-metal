"""ГЕЙТ: файлы данных доезжают в СОБРАННЫЙ пакет, а не только лежат в дереве.

Файл рассчитан быть ОДНИМ И ТЕМ ЖЕ в rwkv-quant и rwkv-metal: пакеты и
файлы данных он находит сам, ничего про конкретный проект не зная. Копии
обязаны совпадать побайтово -- расхождение md5 между деревьями означает,
что кто-то правил одну и забыл вторую.

Написан после 04.09, когда выяснилось, что колесо rwkv-quant не содержало
ни одного файла данных: каталог `rwkv_quant/data/` пакетом не является,
`packages.find` его не брал, а `package-data` и `MANIFEST.in` в проекте
отсутствовали. При этом всё работало у любого, кто запускает из дерева
исходников: путь считается от `__file__`, и файл лежал рядом. Отказ был
виден ТОЛЬКО снаружи дерева -- у того, кто поставил пакет. Цена молчания
измерена: умолчательный путь `quantize(act_stats="auto")` читает
`calib_corpus.txt`, то есть публичный API падал у каждого установившего.

Проверяются оба канала поставки: механизмы включения у них разные, и
починка одного не доказывает починку второго.

  1. Колесо (`pip wheel`) содержит КАЖДЫЙ файл данных из каждого пакета.
  2. Содержимое совпадает с исходником побайтово, а не только по имени.
  3. sdist (`python -m build --sdist`) содержит их же.

Если файлов данных не нашлось вовсе, гейт КРАСНЕЕТ, а не отчитывается
зелёным, ничего не проверив (закон 31).

СОБИРАТЬ ТОЛЬКО ИЗ ЧИСТОЙ КОПИИ ДЕРЕВА (закон 40). Первая редакция гейта
собирала прямо в рабочем каталоге и на мутации -- возврате прежнего
`pyproject.toml` -- осталась ЗЕЛЁНОЙ: `build/lib` и `egg-info/SOURCES.txt`
переживают правку конфигурации, и setuptools их переиспользует, а
`pip --no-cache-dir` снимает не тот кеш. Гейт мерил остатки инструмента.

    python tests/test_packaging_data.py
"""
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {"__pycache__", "build", "dist", "tests"}
SKIP_EXT = (".py", ".pyc", ".pyo", ".so")


def _packages():
    """Каталоги верхнего уровня, являющиеся пакетами."""
    out = []
    for name in sorted(os.listdir(ROOT)):
        d = os.path.join(ROOT, name)
        if name in SKIP_DIRS or name.startswith("."):
            continue
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "__init__.py")):
            out.append(name)
    return out


def _data_files():
    """Пути файлов данных относительно корня дерева.

    Точечные файлы пропускаются намеренно: .DS_Store и подобный мусор в
    пакет ехать НЕ должен, и требовать его наличия в колесе было бы
    ошибкой ровно обратного знака.
    """
    out = []
    for pkg in _packages():
        for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, pkg)):
            dirnames[:] = [d for d in dirnames
                           if d not in SKIP_DIRS and not d.startswith(".")]
            for f in sorted(filenames):
                if f.startswith(".") or f.endswith(SKIP_EXT):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, f), ROOT)
                out.append(rel.replace(os.sep, "/"))
    return sorted(out)


def _clean_copy(dst):
    """Чистая копия дерева для сборки (закон 40).

    Копируем ТО, ЧТО ПОД КОНТРОЛЕМ ВЕРСИЙ. Рабочее дерево может содержать
    гигабайты локальных артефактов -- в rwkv-metal это runs/ на 2.6 ГБ и
    веса на 700 МБ, -- и тащить их ради сборки на пару мегабайт кода
    бессмысленно. Побочно это ещё и проверка: файл данных, не попавший
    под git, не доедет ни до sdist, ни до чужой машины, и гейт обязан на
    нём краснеть, а не молчать.
    """
    r = subprocess.run(["git", "-C", ROOT, "ls-files", "-z"],
                       capture_output=True)
    if r.returncode != 0:
        shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns(
            ".git", "build", "dist", "*.egg-info", "__pycache__",
            "*.pyc", ".venv"), dirs_exist_ok=True)
        return "copytree (не git-дерево)"
    rels = [x.decode("utf-8") for x in r.stdout.split(b"\0") if x]
    n = 0
    for rel in rels:
        src = os.path.join(ROOT, rel)
        if not os.path.isfile(src):
            continue
        out = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copy2(src, out)
        n += 1
    return "git ls-files, %d файлов" % n


def main():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print("  %-46s %s %s" % (name, "ок" if cond else "ПРОВАЛ", detail))

    print("дерево:", ROOT)
    print("пакеты:", ", ".join(_packages()) or "(нет)")
    files = _data_files()
    check("файлы данных в дереве найдены", len(files) > 0, str(files))
    if not files:
        print("\n[FAIL] упаковка данных")
        return False

    with tempfile.TemporaryDirectory() as td:
        # Чистая копия: см. докстринг и закон 40. Без неё мутация не краснеет.
        srcdir = os.path.join(td, "src")
        os.makedirs(srcdir, exist_ok=True)
        print("копия:", _clean_copy(srcdir))

        r = subprocess.run(
            [sys.executable, "-m", "pip", "wheel", srcdir, "--no-deps",
             "--no-build-isolation", "--no-cache-dir", "-q", "-w", td],
            capture_output=True, text=True)
        check("колесо собралось", r.returncode == 0, r.stderr.strip()[-160:])
        whls = [f for f in os.listdir(td) if f.endswith(".whl")]
        check("колесо на месте", len(whls) == 1, str(whls))
        if whls:
            with zipfile.ZipFile(os.path.join(td, whls[0])) as z:
                names = set(z.namelist())
                for rel in files:
                    here = rel in names
                    check("в колесе: %s" % rel, here)
                    if here:
                        want = open(os.path.join(ROOT, rel), "rb").read()
                        got = z.read(rel)
                        check("  байт-в-байт: %s" % rel, got == want,
                              "%d байт" % len(got))

        # sdist -- ВТОРОЙ канал поставки, механизм включения у него свой.
        r2 = subprocess.run(
            [sys.executable, "-m", "build", "--sdist", "--no-isolation",
             "-o", td, srcdir], capture_output=True, text=True)
        if r2.returncode != 0:
            check("sdist собрался", False, r2.stderr.strip()[-160:])
        else:
            tars = [f for f in os.listdir(td) if f.endswith(".tar.gz")]
            check("sdist на месте", len(tars) == 1, str(tars))
            if tars:
                with tarfile.open(os.path.join(td, tars[0])) as t:
                    inner = set("/".join(n.split("/")[1:])
                                for n in t.getnames())
                    for rel in files:
                        check("в sdist: %s" % rel, rel in inner)

    print("\n[%s] упаковка данных" % ("OK" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
