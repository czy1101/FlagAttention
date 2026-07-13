%global debug_package %{nil}

# Distros that ship pyproject-rpm-macros (Fedora, EL9+) build via the
# %%pyproject_* macro family — that path is unchanged. Distros without
# it (openEuler 24.03, EL8-family) fall back to a plain pip
# wheel/install build. Capability-detected at parse time, so the build
# container must have its python toolchain installed before rpmbuild
# runs (both Dockerfile.rpm paths do).
%if %{defined pyproject_wheel}
%global has_pyproject_macros 1
%else
%global has_pyproject_macros 0
%endif

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
# (spec %%build / %%install pass SETUPTOOLS_SCM_PRETEND_VERSION=%%{version} so
#  they self-update with the Version: line above. %% escaped: openEuler's
#  rpm expands macros even in comments and a bare %%install here would
#  terminate the preamble.)
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
BuildRequires:  python3-setuptools_scm
%if %{has_pyproject_macros}
BuildRequires:  pyproject-rpm-macros
%endif

%description
Collection of memory-efficient attention operators implemented in the Triton language, for large language model training and inference.

%prep
%autosetup -n flag-attention-%{version}

%build
export SETUPTOOLS_SCM_PRETEND_VERSION=%{version}
%if %{has_pyproject_macros}
%pyproject_wheel
%else
%{__python3} -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
%endif

%install
export SETUPTOOLS_SCM_PRETEND_VERSION=%{version}
%if %{has_pyproject_macros}
%pyproject_install
%pyproject_save_files flag_attn
%else
%{__python3} -m pip install --no-deps --no-index --no-warn-script-location \
    --root %{buildroot} dist/*.whl
%endif

%check
# Smoke find_spec test (no actual import) — verifies the built module
# lands at the expected sitelib path. Doesn't import the module so
# missing runtime deps (torch, triton, ...) don't trip the check;
# those are user-install-time concerns, not packaging concerns.
PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=%{buildroot}%{python3_sitelib} \
    python3 -c "import importlib.util; s = importlib.util.find_spec('flag_attn'); assert s and s.origin, 'flag_attn not findable'; print('OK: flag_attn at', s.origin)"

%if %{has_pyproject_macros}
%files -f %{pyproject_files}
%license LICENSE
%else
%files
%license LICENSE
%{python3_sitelib}/flag_attn/
%{python3_sitelib}/flag_attn-%{version}.dist-info/
%endif

%changelog
* Mon Jul 13 2026 FlagOS Contributors <contact@flagos.io> - 0.3.0-1
- Add pip-based fallback for distros without pyproject-rpm-macros
  (openEuler 24.03); Fedora build path unchanged.

* Wed May 13 2026 FlagOS Contributors <contact@flagos.io> - 0.3.0-1
- Initial RPM packaging.
