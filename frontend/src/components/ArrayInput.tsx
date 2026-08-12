import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export function ArrayInput({
  label,
  values,
  onChange,
}: {
  label: string;
  values: string[];
  onChange: (vals: string[]) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium">{label}</label>
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-2 text-xs"
          onClick={() => onChange([...values, ""])}
        >
          + 추가
        </Button>
      </div>
      <div className="space-y-1">
        {values.map((val, i) => (
          <div key={i} className="flex gap-1">
            <Input
              value={val}
              onChange={(e) => {
                const next = [...values];
                next[i] = e.target.value;
                onChange(next);
              }}
              className="flex-1 h-8 text-sm"
            />
            <Button
              variant="ghost"
              size="sm"
              className="h-8 w-8 p-0 text-muted-foreground hover:text-destructive"
              onClick={() => onChange(values.filter((_, j) => j !== i))}
            >
              ×
            </Button>
          </div>
        ))}
      </div>
    </div>
  );
}
