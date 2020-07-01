SETLOCAL ENABLEDELAYEDEXPANSION
set /a count = 0
for /d %%v in (../datasets/ShapeNetCore/bench/*) do (
    set /a count +=1
   
    copy ..\datasets\ShapeNetCore\bench\%%v\model.obj ..\data\bench\!count!\
)

ENDLOCAL