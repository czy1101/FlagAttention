%global debug_package %{nil}

# Filter the auto-generated Requires for: triton.
# Reason: distro triton has no current version; users install via pip.
# See packaging/INSTALL.md (or future flagos-packaging install docs) for the
# user-side pip install incantation.
%global __requires_exclude ^python3(\.[0-9]+)?dist\((triton)\)$
Name:           python3-flag-attention
# NOTE: version is duplicated across 4 places — keep them in sync when bumping:
#   1. this Version: line
#   2. packaging/debian/changelog (latest entry)
#   3. packaging/debian/rules SETUPTOOLS_SCM_PRETEND_VERSION
# (spec %build / %install pass SETUPTOOLS_SCM_PRETEND_VERSION=%{version} so
#  they self-update with the Version: line above.)
Version:        0.3.0
Release:        1%{?dist}
Summary:        FlagAttention — memory-efficient attention operators (Triton)

License:        Apache-2.0
URL:            https://github.com/flagos-ai/FlagAttention
Source0:        %{url}/archive/refs/tags/v%{version}.tar.gz#/flag-attention-%{version}.tar.gz
BuildArch:      noarch
BuildRequires:  python3-devel
BuildRequires:  python3-setuptools >= 60
BuildRequires:  python3-wheel
BuildRequires:  python3-pip
BuildRequires:  pyproject-rpm-macros
BuildRequires:  python3-setuptools_scm

%description
Collection of memory-efficient attention operators implemented in the Triton language, for large language model training and inference.

%prep
%autosetup -n flag-attention-%{version}

%build
export SETUPTOOLS_SCM_PRETEND_VERSION=%{version}
%pyproject_wheel

%install
export SETUPTOOLS_SCM_PRETEND_VERSION=%{version}
%pyproject_install
%pyproject_save_files flag_attn

%check
# Smoke find_spec test (no actual import) — verifies the built module
# lands at the expected sitelib path. Doesn't import the module so
# missing runtime deps (torch, triton, ...) don't trip the check;
# those are user-install-time concerns, not packaging concerns.
PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=%{buildroot}%{python3_sitelib} \
    python3 -c "import importlib.util; s = importlib.util.find_spec('flag_attn'); assert s and s.origin, 'flag_attn not findable'; print('OK: flag_attn at', s.origin)"
# Subpackage completeness: compare the __init__.py sets between src/ and
# the installed buildroot. Catches packages.find configs that silently
# drop subpackages (same check as in packaging/debian/rules). Filesystem
# check on purpose: find_spec on a subpackage would import
# flag_attn/__init__.py, which needs torch/triton — not available (and
# not wanted) in the build environment.
src_inits="$(cd src && find flag_attn -name __init__.py)"
if [ -z "$src_inits" ]; then
    echo "ERROR: no __init__.py found under src/flag_attn; check would be vacuous"
    exit 1
fi
missing=""
for f in $src_inits; do
    if [ ! -f "%{buildroot}%{python3_sitelib}/$f" ]; then missing="$missing $f"; fi
done
if [ -n "$missing" ]; then
    echo "ERROR: subpackages missing from buildroot:$missing"
    exit 1
fi
echo "OK: all package __init__.py files from src/ present in buildroot"

%files -f %{pyproject_files}
%license LICENSE

%changelog
* Wed May 13 2026 FlagOS Contributors <contact@flagos.io> - 0.3.0-1
- Initial RPM packaging.
