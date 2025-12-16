{
  description = "AI development shell with Python, JupyterLab, FFmpeg, and tooling";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = {nixpkgs, ...}:
  let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
  in {
    devShells.${system}.default = pkgs.mkShell {
      nativeBuildInputs = with pkgs; [
        gcc
        jellyfin-ffmpeg
        neovim
        nushell
        starship
        python312Packages.tkinter
        python312Packages.pandas
        python312Packages.stdenv
        python312
        python312Packages.opencv4
        python312Packages.numba
        python312Packages.pip
        python312Packages.flake8
        python312Packages.black
        python312Packages.z3-solver
        python312Packages.jupyterlab
        python312Packages.uv
      ];
      shellHook = ''
        export SHELL=$(which nu)
        export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib/
        echo "Welcome to the AI development environment"
      '';
    };
  };
}



