import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/** Catches a render/runtime error anywhere below and shows the message instead of blanking the
 *  window - the engine's errors are the interesting ones. */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("Corpus Studio render error:", error, info.componentStack);
  }

  render(): ReactNode {
    if (this.state.error) {
      return (
        <div className="cs-body">
          <div className="cs-error" role="alert">
            <strong>Something broke.</strong> {this.state.error.message}
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
