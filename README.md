# README — Setup no Windows (Python 3.12), Gurobi e HiGHS (MIP) e CBC

Guia objetivo em **PowerShell**. Execute os blocos na ordem.

---

## 1) Instalar Python 3.12 e criar o ambiente virtual

```powershell
# Baixar instalador (marque "Add python.exe to PATH" na instalação)
Start-Process "https://www.python.org/downloads/release/python-31210/"
```

- Após instalar, crie o venv
```powershell
py -3.12 -m venv .venv312
```

- Ativar o venv (Windows)
```powershell
.\.venv312\Scripts\activate
```

- (opcional) Confirmar versão
```powershell
python --version
```

---

## 2) Microsoft Tools (se solicitado por pacotes que compilam C/C++)
Instale os componentes:
- MSVC v14.x (toolset v143 ou superior)
- Windows 10/11 SDK
- C++ CMake tools for Windows

## 3) Instalar dependências do projeto
Com o venv (.venv312) ativado e no diretório do projeto:
```powershell
pip install -r requirements.txt
```

## 4) Rodar a aplicação
```powershell
python modelo.py
```

---

# FAZER PROCEDIMENTO ABAIXO COM O SOLVER HIGHS CASO SEJA NECESSÁRIO

## 1) Baixar e compilar o HiGHS (Buildar via Git) (gera DLL)
Pré-requisito: CMake instalado e no PATH
https://cmake.org/download/

Clonar o repositório
```powershell
git clone https://github.com/ERGO-Code/HiGHS.git
cd HiGHS
```

Gerar build compartilhado (DLL) a partir da RAIZ do repositório
```powershell
rmdir .\build-shared -Recurse -Force -ErrorAction SilentlyContinue

cmake -S . -B build-shared -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_BUILD_TYPE=Release `
  -DBUILD_SHARED_LIBS=ON `
  -DHIGHS_BUILD_TESTS=OFF `
  -DHIGHS_BUILD_EXAMPLES=OFF

cmake --build build-shared --config Release --target highs
```

A DLL normalmente fica em: .\HiGHS\build-shared\bin\Release\highs.dll

## 2) Disponibilizar a DLL para o Python-MIP

```powershell
[Environment]::SetEnvironmentVariable(
  'PMIP_HIGHS_LIBRARY',
  'C:\Users\thiago.assis\.vscode\HIGHS\HiGHS\build-shared\bin\Release\highs.dll',
  'User'
)
$env:PMIP_HIGHS_LIBRARY = 'C:\Users\thiago.assis\.vscode\HIGHS\HiGHS\build-shared\bin\Release\highs.dll'
```

Importante: o Python-MIP precisa de highs.dll (biblioteca dinâmica).
Arquivo .lib não funciona para o carregamento via Python.
