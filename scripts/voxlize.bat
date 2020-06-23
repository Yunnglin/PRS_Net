SETLOCAL ENABLEDELAYEDEXPANSION
set /a count = 0
set /a size = 32
for /d %%v in (../datasets/ShapeNetCore/skateboard/04225987/*) do (
    set /a count +=1
    md ..\data\!count!
    binvox.exe -d %size% -t nrrd ..\datasets\ShapeNetCore\skateboard\04225987\%%v\model.obj
    move ..\datasets\ShapeNetCore\skateboard\04225987\%%v\model.nrrd ..\data\!count!
)

ENDLOCAL