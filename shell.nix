{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = [
    pkgs.python3
    pkgs.python3Packages.numpy
    pkgs.python3Packages.pandas
    pkgs.python3Packages.torch
    pkgs.python3Packages.gpytorch
    pkgs.uv
  ];
  
  # Set PYTHONPATH to include the current directory so we can import bioprocess_gp
  shellHook = ''
    export PYTHONPATH=$PYTHONPATH:.
  '';
}
