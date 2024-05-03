%%º¯ÊıÓ³Éä%%
function y_turn = mapping(y,y_old_min,y_old_max,y_new_min,y_new_max)
    temp = (y_new_max-y_new_min)/(y_old_max-y_old_min);
    y_turn = (y-y_old_min)*temp+y_new_min;
end

